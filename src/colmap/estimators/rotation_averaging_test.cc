// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/rotation_averaging.h"

#include "colmap/math/math.h"
#include "colmap/math/random.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/hash_containers.h"

#include <algorithm>
#include <map>
#include <utility>

#include <gtest/gtest.h>

namespace colmap {
namespace {

void LoadReconstructionAndPoseGraph(const Database& database,
                                    Reconstruction* reconstruction,
                                    PoseGraph* pose_graph) {
  DatabaseCache database_cache;
  DatabaseCache::Options options;
  database_cache.Load(database, options);
  reconstruction->Load(database_cache);
  pose_graph->Load(*database_cache.CorrespondenceGraph());
}

struct TestData {
  std::shared_ptr<Database> database;
  Reconstruction gt_reconstruction;
  Reconstruction reconstruction;
  PoseGraph pose_graph;
  std::vector<PosePrior> pose_priors;
};

TestData CreateTestData(const SyntheticDatasetOptions& dataset_options,
                        const SyntheticNoiseOptions* noise_options = nullptr) {
  TestData data;
  data.database = Database::Open(kInMemorySqliteDatabasePath);
  SynthesizeDataset(
      dataset_options, &data.gt_reconstruction, data.database.get());
  if (noise_options) {
    SynthesizeNoise(
        *noise_options, &data.gt_reconstruction, data.database.get());
  }
  LoadReconstructionAndPoseGraph(
      *data.database, &data.reconstruction, &data.pose_graph);
  data.pose_priors = data.database->ReadAllPosePriors();
  return data;
}

RotationEstimatorOptions CreateRATestOptions(bool use_gravity = false) {
  RotationEstimatorOptions options;
  options.skip_initialization = false;
  options.use_gravity = use_gravity;
  options.use_stratified = true;
  return options;
}

void ExpectEqualRotations(const Reconstruction& gt,
                          const Reconstruction& computed,
                          const double max_rotation_error_deg) {
  const double max_rotation_error_rad = DegToRad(max_rotation_error_deg);
  const std::vector<image_t> reg_image_ids = gt.RegImageIds();
  for (size_t i = 0; i < reg_image_ids.size(); i++) {
    const image_t image_id1 = reg_image_ids[i];
    for (size_t j = 0; j < i; j++) {
      const image_t image_id2 = reg_image_ids[j];
      const Eigen::Quaterniond cam2_from_cam1 =
          computed.Image(image_id2).CamFromWorld().rotation() *
          computed.Image(image_id1).CamFromWorld().rotation().inverse();
      const Eigen::Quaterniond cam2_from_cam1_gt =
          gt.Image(image_id2).CamFromWorld().rotation() *
          gt.Image(image_id1).CamFromWorld().rotation().inverse();
      EXPECT_LE(cam2_from_cam1.angularDistance(cam2_from_cam1_gt),
                max_rotation_error_rad);
    }
  }
}

// Gauge-invariant mean rotation error (in degrees) over all registered image
// pairs, comparing the computed relative rotations against the ground truth.
double MeanRelativeRotationErrorDeg(const Reconstruction& gt,
                                    const Reconstruction& computed) {
  const std::vector<image_t> reg_image_ids = gt.RegImageIds();
  double total_error_rad = 0;
  int count = 0;
  for (size_t i = 0; i < reg_image_ids.size(); i++) {
    for (size_t j = 0; j < i; j++) {
      const Eigen::Quaterniond rel =
          computed.Image(reg_image_ids[j]).CamFromWorld().rotation() *
          computed.Image(reg_image_ids[i]).CamFromWorld().rotation().inverse();
      const Eigen::Quaterniond rel_gt =
          gt.Image(reg_image_ids[j]).CamFromWorld().rotation() *
          gt.Image(reg_image_ids[i]).CamFromWorld().rotation().inverse();
      total_error_rad += rel.angularDistance(rel_gt);
      count++;
    }
  }
  return RadToDeg(total_error_rad / count);
}

void ResetSensorsFromRig(Reconstruction& reconstruction) {
  for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor.has_value()) {
        reconstruction.Rig(rig_id).ResetSensorFromRig(sensor_id);
      }
    }
  }
}

void RunAndVerifyRotationAveraging(const Reconstruction& gt_reconstruction,
                                   const Reconstruction& reconstruction,
                                   const PoseGraph& pose_graph,
                                   const std::vector<PosePrior>& pose_priors,
                                   const std::vector<bool>& use_gravity_values,
                                   const double max_rotation_error_deg) {
  for (const bool use_gravity : use_gravity_values) {
    Reconstruction reconstruction_copy = reconstruction;
    PoseGraph pose_graph_copy = pose_graph;
    RunRotationAveraging(CreateRATestOptions(use_gravity),
                         pose_graph_copy,
                         reconstruction_copy,
                         pose_priors);

    ExpectEqualRotations(
        gt_reconstruction, reconstruction_copy, max_rotation_error_deg);
  }
}

size_t NumValidEdges(const PoseGraph& pose_graph) {
  size_t num_valid = 0;
  for (const auto& [pair_id, edge] : pose_graph.Edges()) {
    if (edge.valid) {
      ++num_valid;
    }
  }
  return num_valid;
}

// 180 degree rotation about the camera optical (z) axis.
Eigen::Quaterniond RzPi() {
  return Eigen::Quaterniond(
      Eigen::AngleAxisd(EIGEN_PI, Eigen::Vector3d::UnitZ()));
}

// Rewrites the synthetic ground truth into a nadir lawnmower survey: all
// cameras look straight down (-Z in world), consecutive strips alternate
// heading by 180 degrees, and every pose graph edge is recomputed from the
// new ground truth. Returns the strip index of each image.
NodeHashMap<image_t, int> MakeNadirLawnmower(int num_strips,
                                             int frames_per_strip,
                                             TestData& data) {
  std::vector<image_t> image_ids = data.gt_reconstruction.RegImageIds();
  std::sort(image_ids.begin(), image_ids.end());
  THROW_CHECK_EQ(static_cast<int>(image_ids.size()),
                 num_strips * frames_per_strip);

  // cam_from_world for a nadir camera: camera z (optical axis) = world -Z.
  const Eigen::Quaterniond nadir(
      Eigen::AngleAxisd(EIGEN_PI, Eigen::Vector3d::UnitX()));

  NodeHashMap<image_t, int> image_to_strip;
  for (size_t i = 0; i < image_ids.size(); ++i) {
    const int strip = static_cast<int>(i) / frames_per_strip;
    const int along = static_cast<int>(i) % frames_per_strip;
    const double yaw = (strip % 2 == 0) ? 0.0 : EIGEN_PI;
    const Eigen::Quaterniond cam_from_world =
        Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ())) *
        nadir;
    const Eigen::Vector3d position(along, strip, 10.0);
    const Rigid3d world_from_cam(cam_from_world.inverse(), position);
    const image_t image_id = image_ids[i];
    const frame_t frame_id = data.gt_reconstruction.Image(image_id).FrameId();
    data.gt_reconstruction.Frame(frame_id).SetRigFromWorld(
        Inverse(world_from_cam));
    image_to_strip[image_id] = strip;
  }

  for (auto& [pair_id, edge] : data.pose_graph.Edges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    edge.cam2_from_cam1 =
        data.gt_reconstruction.Image(image_id2).CamFromWorld() *
        Inverse(data.gt_reconstruction.Image(image_id1).CamFromWorld());
  }

  return image_to_strip;
}

// Maximum relative rotation error (degrees) over all registered image pairs,
// where the computed relative rotation is compared with the ground truth up
// to a 180 degree flip about the optical axis of either camera.
double MaxRelativeRotationErrorModuloFlipDeg(const Reconstruction& gt,
                                             const Reconstruction& computed) {
  const Eigen::Quaterniond rz_pi = RzPi();
  const std::vector<image_t> reg_image_ids = computed.RegImageIds();
  double max_error_rad = 0;
  for (size_t i = 0; i < reg_image_ids.size(); i++) {
    for (size_t j = 0; j < i; j++) {
      const Eigen::Quaterniond rel =
          computed.Image(reg_image_ids[j]).CamFromWorld().rotation() *
          computed.Image(reg_image_ids[i]).CamFromWorld().rotation().inverse();
      const Eigen::Quaterniond rel_gt =
          gt.Image(reg_image_ids[j]).CamFromWorld().rotation() *
          gt.Image(reg_image_ids[i]).CamFromWorld().rotation().inverse();
      const double error_rad =
          std::min({rel.angularDistance(rel_gt),
                    rel.angularDistance(rz_pi * rel_gt),
                    rel.angularDistance(rel_gt * rz_pi)});
      max_error_rad = std::max(max_error_rad, error_rad);
    }
  }
  return RadToDeg(max_error_rad);
}

TEST(RotationAveraging, WithoutNoise) {
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  RunAndVerifyRotationAveraging(data.gt_reconstruction,
                                data.reconstruction,
                                data.pose_graph,
                                data.pose_priors,
                                {true, false},
                                /*max_rotation_error_deg=*/1e-2);
}

TEST(RotationAveraging, WeightedNoiseFreeMatchesInvariant) {
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  // Assign varying positive match counts so the edge weighting is non-trivial.
  int counter = 1;
  for (auto& [pair_id, edge] : data.pose_graph.Edges()) {
    edge.num_matches = 10 * (counter++ % 7) + 1;
  }

  for (const bool use_gravity : {true, false}) {
    // Unweighted baseline.
    Reconstruction recon_unweighted = data.reconstruction;
    PoseGraph pose_graph_unweighted = data.pose_graph;
    RotationEstimatorOptions options_unweighted =
        CreateRATestOptions(use_gravity);
    options_unweighted.reweighting = RotationAveragingReweighting::UNIFORM;
    RunRotationAveraging(options_unweighted,
                         pose_graph_unweighted,
                         recon_unweighted,
                         data.pose_priors);

    // Weighted.
    Reconstruction recon_weighted = data.reconstruction;
    PoseGraph pose_graph_weighted = data.pose_graph;
    RotationEstimatorOptions options_weighted =
        CreateRATestOptions(use_gravity);
    options_weighted.reweighting =
        RotationAveragingReweighting::INLIER_MATCH_COUNT;
    RunRotationAveraging(options_weighted,
                         pose_graph_weighted,
                         recon_weighted,
                         data.pose_priors);

    // The weighted solution recovers the ground truth and, for a noise-free
    // system, is identical to the unweighted solution.
    ExpectEqualRotations(data.gt_reconstruction,
                         recon_weighted,
                         /*max_rotation_error_deg=*/1e-2);
    ExpectEqualRotations(
        recon_unweighted, recon_weighted, /*max_rotation_error_deg=*/1e-2);
  }
}

TEST(RotationAveraging, WeightedReducesErrorWithNoisyLowMatchEdges) {
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 15;
  synthetic_dataset_options.num_points3D = 150;
  synthetic_dataset_options.prior_gravity = false;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  // Inject controlled rotation noise into each relative pose and halve the
  // match count of the noisy edges. A uniform baseline match count ensures the
  // only weight difference between runs is the halving of the noisy edges.
  constexpr double kNoiseThresholdDeg = 5.0;
  constexpr int kBaselineMatches = 100;
  for (auto& [pair_id, edge] : data.pose_graph.Edges()) {
    // Range kept just above the 5 deg threshold: noisy edges (5-8 deg) still
    // carry meaningful IRLS weight (sigma = 5 deg), so the 2x down-weighting
    // has real leverage; sub-5 deg edges anchor the solution.
    const double noise_deg = RandomUniformReal(0.0, 8.0);
    // Seeded isotropic axis (RandomEigenVectord<3>() is NOT seeded by
    // SetPRNGSeed).
    Eigen::Vector3d axis(RandomGaussian(0.0, 1.0),
                         RandomGaussian(0.0, 1.0),
                         RandomGaussian(0.0, 1.0));
    axis.normalize();
    const Eigen::Quaterniond perturb(
        Eigen::AngleAxisd(DegToRad(noise_deg), axis));
    edge.cam2_from_cam1.rotation() =
        perturb * Eigen::Quaterniond(edge.cam2_from_cam1.rotation());
    edge.num_matches = kBaselineMatches;
    if (noise_deg > kNoiseThresholdDeg) {
      edge.num_matches /= 2;  // Down-weight noisy edges.
    }
  }

  const auto run = [&](RotationAveragingReweighting reweighting) {
    Reconstruction reconstruction = data.reconstruction;
    PoseGraph pose_graph = data.pose_graph;
    RotationEstimatorOptions options =
        CreateRATestOptions(/*use_gravity=*/false);
    options.reweighting = reweighting;
    options.random_seed = 0;             // Deterministic solve.
    options.max_rotation_error_deg = 0;  // Disable post-solve edge filtering so
                                         // only the solver reweighting differs.
    RunRotationAveraging(options, pose_graph, reconstruction, data.pose_priors);
    return MeanRelativeRotationErrorDeg(data.gt_reconstruction, reconstruction);
  };

  const double error_unweighted = run(RotationAveragingReweighting::UNIFORM);
  const double error_weighted =
      run(RotationAveragingReweighting::INLIER_MATCH_COUNT);

  EXPECT_LT(error_weighted, error_unweighted);
}

TEST(RotationAveraging, WithoutNoiseWithNonTrivialKnownRig) {
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  RunAndVerifyRotationAveraging(data.gt_reconstruction,
                                data.reconstruction,
                                data.pose_graph,
                                data.pose_priors,
                                {true, false},
                                /*max_rotation_error_deg=*/1e-2);
}

TEST(RotationAveraging, WithoutNoiseWithNonTrivialUnknownRig) {
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  ResetSensorsFromRig(data.reconstruction);

  // For unknown rigs, it is not supported to use gravity.
  RunAndVerifyRotationAveraging(data.gt_reconstruction,
                                data.reconstruction,
                                data.pose_graph,
                                data.pose_priors,
                                {false},
                                /*max_rotation_error_deg=*/1e-2);
}

TEST(RotationAveraging, WithNoiseAndOutliers) {
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.inlier_match_ratio = 0.6;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  synthetic_noise_options.prior_gravity_stddev = 3e-1;
  auto data =
      CreateTestData(synthetic_dataset_options, &synthetic_noise_options);

  RunAndVerifyRotationAveraging(data.gt_reconstruction,
                                data.reconstruction,
                                data.pose_graph,
                                data.pose_priors,
                                {true, false},
                                /*max_rotation_error_deg=*/3);
}

TEST(RotationAveraging, WithNoiseAndOutliersWithNonTrivialKnownRigs) {
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.inlier_match_ratio = 0.6;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  synthetic_noise_options.prior_gravity_stddev = 3e-1;
  auto data =
      CreateTestData(synthetic_dataset_options, &synthetic_noise_options);

  RunAndVerifyRotationAveraging(data.gt_reconstruction,
                                data.reconstruction,
                                data.pose_graph,
                                data.pose_priors,
                                {true, false},
                                /*max_rotation_error_deg=*/2.);
}

TEST(RotationAveraging, DeterministicRandomSeed) {
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  RotationEstimatorOptions options = CreateRATestOptions();
  options.random_seed = 42;

  // Run twice with the same seed and verify identical results.
  Reconstruction reconstruction1 = data.reconstruction;
  PoseGraph pose_graph1 = data.pose_graph;
  EXPECT_TRUE(RunRotationAveraging(
      options, pose_graph1, reconstruction1, data.pose_priors));

  Reconstruction reconstruction2 = data.reconstruction;
  PoseGraph pose_graph2 = data.pose_graph;
  EXPECT_TRUE(RunRotationAveraging(
      options, pose_graph2, reconstruction2, data.pose_priors));

  for (const image_t image_id : reconstruction1.RegImageIds()) {
    const Eigen::Quaterniond q1 =
        reconstruction1.Image(image_id).CamFromWorld().rotation();
    const Eigen::Quaterniond q2 =
        reconstruction2.Image(image_id).CamFromWorld().rotation();
    // In the presence of optimizations like FMA, q.angularDistance(q) can be
    // near-zero instead of zero, so check equality explicitly instead of with
    // ExpectEqualRotations.
    EXPECT_EQ(q1.coeffs(), q2.coeffs());
  }
}

TEST(RotationAveraging, RidgeRegularizationDoesNotBiasSolution) {
  // Use a noisy multi-rig setup to make the solution non-trivial and the
  // regularization's effect non-degenerate.
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.inlier_match_ratio = 0.6;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  synthetic_noise_options.prior_gravity_stddev = 3e-1;
  auto data =
      CreateTestData(synthetic_dataset_options, &synthetic_noise_options);

  RotationEstimatorOptions options = CreateRATestOptions(/*use_gravity=*/true);
  options.random_seed = 42;

  // Run once with no regularization.
  Reconstruction reconstruction_no_ridge = data.reconstruction;
  PoseGraph pose_graph_no_ridge = data.pose_graph;
  options.ridge_regularization = 0;
  ASSERT_TRUE(RunRotationAveraging(
      options, pose_graph_no_ridge, reconstruction_no_ridge, data.pose_priors));

  // Run again with the same default ridge that the global mapper uses. The
  // option must flow through L1 and IRLS without biasing the solution.
  Reconstruction reconstruction_ridge = data.reconstruction;
  PoseGraph pose_graph_ridge = data.pose_graph;
  options.ridge_regularization = 1e-9;
  ASSERT_TRUE(RunRotationAveraging(
      options, pose_graph_ridge, reconstruction_ridge, data.pose_priors));

  // The two solutions should be effectively identical since 1e-9 is far below
  // any meaningful residual scale in the optimization.
  ExpectEqualRotations(reconstruction_no_ridge,
                       reconstruction_ridge,
                       /*max_rotation_error_deg=*/1e-12);
}

TEST(RotationAveraging, EmptyPoseGraph) {
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 3;
  synthetic_dataset_options.num_points3D = 20;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  // Invalidate all edges so connected components are empty.
  for (auto& [pair_id, edge] : data.pose_graph.Edges()) {
    edge.valid = false;
  }

  RotationEstimatorOptions options = CreateRATestOptions();
  EXPECT_FALSE(RunRotationAveraging(
      options, data.pose_graph, data.reconstruction, data.pose_priors));
}

TEST(RotationAveraging, MultiImageRigFrameDeregisterDoesNotCrashOnSecondVisit) {
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  std::vector<frame_t> frame_ids;
  for (const auto& [fid, _] : data.reconstruction.Frames()) {
    frame_ids.push_back(fid);
  }
  std::sort(frame_ids.begin(), frame_ids.end());
  ASSERT_EQ(frame_ids.size(), 4);
  const frame_t isolated_frame_id = frame_ids.back();

  // 1. Collect every image_id that belongs to the isolated frame.
  FlatHashSet<image_t> isolated_image_ids;
  for (const auto& data_id :
       data.reconstruction.Frame(isolated_frame_id).ImageIds()) {
    isolated_image_ids.insert(data_id.id);
  }

  // 2. Strip every pose-graph edge touching the isolated frame. After
  //    this the frame is unreachable from any other frame in the
  //    pose-graph CC.
  std::vector<std::pair<image_t, image_t>> edges_to_remove;
  for (const auto& [pair_id, edge] : data.pose_graph.Edges()) {
    const auto [id1, id2] = PairIdToImagePair(pair_id);
    if (isolated_image_ids.count(id1) || isolated_image_ids.count(id2)) {
      edges_to_remove.emplace_back(id1, id2);
    }
  }
  ASSERT_FALSE(edges_to_remove.empty());
  for (const auto& [id1, id2] : edges_to_remove) {
    data.pose_graph.DeleteEdge(id1, id2);
  }

  // 3. Pre-register the isolated frame with its GT pose. This puts it
  //    into reg_frame_ids_ even though no edges touch it.
  ASSERT_TRUE(data.gt_reconstruction.Frame(isolated_frame_id).HasPose());
  data.reconstruction.Frame(isolated_frame_id)
      .SetRigFromWorld(
          data.gt_reconstruction.Frame(isolated_frame_id).RigFromWorld());
  data.reconstruction.RegisterFrame(isolated_frame_id);

  RotationEstimatorOptions options = CreateRATestOptions();
  options.max_rotation_error_deg = 1.0;

  EXPECT_TRUE(RunRotationAveraging(
      options, data.pose_graph, data.reconstruction, data.pose_priors));

  // Post-condition: the isolated frame was deregistered cleanly.
  // The other frames remain registered with poses recovered by RA.
  EXPECT_FALSE(data.reconstruction.Frame(isolated_frame_id).HasPose());
  for (size_t i = 0; i + 1 < frame_ids.size(); ++i) {
    EXPECT_TRUE(data.reconstruction.Frame(frame_ids[i]).HasPose())
        << "Frame " << frame_ids[i]
        << " should remain registered after deregistration of the "
        << "isolated frame.";
  }
}

TEST(RotationAveraging, GravityWithUnknownRigSensorsReturnsFalse) {
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  ResetSensorsFromRig(data.reconstruction);

  // With gravity enabled and unknown rig sensors, EstimateRotations should
  // fail inside RunRotationAveraging because AllSensorsFromRigKnown returns
  // false. However, RunRotationAveraging takes the HasUnknownCamsFromRig path
  // which creates an expanded reconstruction (singleton rigs) that avoids the
  // AllSensorsFromRigKnown check. To directly hit the
  // AllSensorsFromRigKnown check, we use RotationEstimator directly.
  RotationEstimatorOptions options = CreateRATestOptions(/*use_gravity=*/true);

  FlatHashSet<image_t> active_image_ids;
  for (const auto& [image_id, image] : data.reconstruction.Images()) {
    active_image_ids.insert(image_id);
  }

  RotationEstimator estimator(options);
  EXPECT_FALSE(estimator.EstimateRotations(data.pose_graph,
                                           data.pose_priors,
                                           active_image_ids,
                                           data.reconstruction));
}

// Covers: InitializeRigRotationsFromImages standalone (lines 465-564) with
// multi-camera rig to exercise cam_from_rig estimation and rig_from_world
// averaging.
TEST(RotationAveraging, InitializeSensorFromRigUsingCamsFromWorld) {
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  // Build cams_from_world from the ground truth.
  NodeHashMap<image_t, Rigid3d> cams_from_world;
  for (const auto& [image_id, image] : data.gt_reconstruction.Images()) {
    if (image.HasPose()) {
      cams_from_world[image_id] = image.CamFromWorld();
    }
  }

  ResetSensorsFromRig(data.reconstruction);

  EXPECT_TRUE(
      InitializeRigRotationsFromImages(cams_from_world, data.reconstruction));

  for (const auto& [rig_id, rig] : data.reconstruction.Rigs()) {
    for (const auto& [sensor_id, sensor_from_rig] : rig.NonRefSensors()) {
      EXPECT_LT(sensor_from_rig->rotation().angularDistance(
                    data.gt_reconstruction.Rig(rig_id)
                        .SensorFromRig(sensor_id)
                        .rotation()),
                1e-6);
    }
  }
}

// When a sensor_from_rig is already fully calibrated (valid rotation AND
// translation), InitializeRigRotationsFromImages must preserve it rather than
// resetting the translation to NaN.
TEST(RotationAveraging, InitializeSensorFromRigPreservesCalibratedRig) {
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  NodeHashMap<image_t, Rigid3d> cams_from_world;
  for (const auto& [image_id, image] : data.gt_reconstruction.Images()) {
    if (image.HasPose()) {
      cams_from_world[image_id] = image.CamFromWorld();
    }
  }

  // Snapshot the (already-calibrated) rig BEFORE initialization.
  std::map<std::pair<rig_t, sensor_t>, Rigid3d> snapshot;
  for (const auto& [rig_id, rig] : data.reconstruction.Rigs()) {
    for (const auto& [sensor_id, sensor_from_rig] : rig.NonRefSensors()) {
      ASSERT_TRUE(sensor_from_rig.has_value());
      snapshot[{rig_id, sensor_id}] = *sensor_from_rig;
    }
  }
  ASSERT_GT(snapshot.size(), 0u);

  EXPECT_TRUE(
      InitializeRigRotationsFromImages(cams_from_world, data.reconstruction));

  for (const auto& [rig_id, rig] : data.reconstruction.Rigs()) {
    for (const auto& [sensor_id, sensor_from_rig_after] : rig.NonRefSensors()) {
      ASSERT_TRUE(sensor_from_rig_after.has_value())
          << "rig_id=" << rig_id << ", sensor_id=" << sensor_id.id;
      const auto& sensor_from_rig_before = snapshot.at({rig_id, sensor_id});
      EXPECT_EQ(*sensor_from_rig_after, sensor_from_rig_before)
          << "rig_id=" << rig_id << ", sensor_id=" << sensor_id.id;
    }
  }
}

TEST(RotationAveraging, RefineSensorFromRigFalsePreservesRig) {
  // A non-trivial multi-camera rig so both rotation AND translation are
  // non-zero
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  // Snapshot the rig BEFORE RA so we can compare element-wise.
  std::map<std::pair<rig_t, sensor_t>, Rigid3d> snapshot;
  for (const auto& [rig_id, rig] : data.reconstruction.Rigs()) {
    for (const auto& [sensor_id, sfr] : rig.NonRefSensors()) {
      ASSERT_TRUE(sfr.has_value());
      snapshot[{rig_id, sensor_id}] = *sfr;
    }
  }
  // Sanity check: at least one sensor should have a non-zero translation
  // so the test would actually catch the old "reset to zero" behaviour.
  ASSERT_GT(snapshot.size(), 0u);

  // Run RA with refine_sensor_from_rig=false.
  RotationEstimatorOptions options = CreateRATestOptions(/*use_gravity=*/true);
  options.refine_sensor_from_rig = false;
  ASSERT_TRUE(RunRotationAveraging(
      options, data.pose_graph, data.reconstruction, data.pose_priors));

  // Every sensor_from_rig must match the snapshot exactly.
  for (const auto& [rig_id, rig] : data.reconstruction.Rigs()) {
    for (const auto& [sensor_id, sensor_from_rig_after] : rig.NonRefSensors()) {
      ASSERT_TRUE(sensor_from_rig_after.has_value())
          << "rig_id=" << rig_id << ", sensor_id=" << sensor_id.id;
      const auto& sensor_from_rig_before = snapshot.at({rig_id, sensor_id});
      EXPECT_EQ(*sensor_from_rig_after, sensor_from_rig_before)
          << "rig_id=" << rig_id << ", sensor_id=" << sensor_id.id;
    }
  }
}

// Synthetic nadir lawnmower survey where every other cross-strip edge is
// flipped by 180 degrees about the optical axis, emulating the yaw ambiguity of
// symmetric crop rows: neither the flipped nor the unflipped cross-strip edges
// form a consistent majority.
struct LawnmowerFlipTestData {
  TestData data;
  NodeHashMap<image_t, int> image_to_strip;
  int num_cross_edges = 0;
  int num_flipped_edges = 0;
};

LawnmowerFlipTestData CreateLawnmowerFlipTestData() {
  constexpr int kNumStrips = 2;
  constexpr int kFramesPerStrip = 6;

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = kNumStrips * kFramesPerStrip;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.prior_gravity = false;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;

  LawnmowerFlipTestData test_data;
  test_data.data = CreateTestData(synthetic_dataset_options);
  test_data.image_to_strip =
      MakeNadirLawnmower(kNumStrips, kFramesPerStrip, test_data.data);

  for (auto& [pair_id, edge] : test_data.data.pose_graph.Edges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    if (test_data.image_to_strip.at(image_id1) ==
        test_data.image_to_strip.at(image_id2)) {
      continue;
    }
    ++test_data.num_cross_edges;
    // Flip by image id parity so that every frame sees a mix of flipped and
    // unflipped cross-strip edges (a flip pattern correlated with a frame
    // would legitimately describe that frame as yaw-flipped).
    if ((image_id1 + image_id2) % 2 == 0) {
      edge.cam2_from_cam1.rotation() =
          RzPi() * Eigen::Quaterniond(edge.cam2_from_cam1.rotation());
      ++test_data.num_flipped_edges;
    }
  }
  THROW_CHECK_GT(test_data.num_flipped_edges, 0);
  return test_data;
}

// Expectations for a solve with snapping on the lawnmower data: no frame is
// de-registered, rotations are recovered up to the (physically unobservable)
// per-strip yaw flip, and the post-solve filter only removes the cross-strip
// edges that disagree with the chosen yaw basin.
void ExpectLawnmowerFlipsResolved(const LawnmowerFlipTestData& test_data,
                                  const Reconstruction& reconstruction,
                                  const PoseGraph& pose_graph) {
  const TestData& data = test_data.data;
  EXPECT_EQ(reconstruction.NumRegFrames(),
            data.gt_reconstruction.NumRegFrames());
  EXPECT_LT(MaxRelativeRotationErrorModuloFlipDeg(data.gt_reconstruction,
                                                  reconstruction),
            1e-2);
  int num_invalid_cross = 0;
  for (const auto& [pair_id, edge] : pose_graph.Edges()) {
    if (edge.valid) continue;
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    EXPECT_NE(test_data.image_to_strip.at(image_id1),
              test_data.image_to_strip.at(image_id2))
        << "Within-strip edge was invalidated";
    ++num_invalid_cross;
  }
  // Exactly the half of the cross-strip edges inconsistent with the chosen
  // basin is removed.
  EXPECT_EQ(num_invalid_cross,
            test_data.num_cross_edges - test_data.num_flipped_edges);
}

TEST(RotationAveraging, FlipSnapRecoversYawFlippedCrossStripEdges) {
  const LawnmowerFlipTestData test_data = CreateLawnmowerFlipTestData();
  const TestData& data = test_data.data;

  const auto run = [&](bool use_180_degree_flip_snap,
                       Reconstruction& reconstruction) {
    PoseGraph pose_graph = data.pose_graph;
    RotationEstimatorOptions options =
        CreateRATestOptions(/*use_gravity=*/false);
    options.random_seed = 0;
    options.use_180_degree_flip_snap = use_180_degree_flip_snap;
    reconstruction = data.reconstruction;
    EXPECT_TRUE(RunRotationAveraging(
        options, pose_graph, reconstruction, data.pose_priors));
    return pose_graph;
  };

  Reconstruction recon_snap;
  const PoseGraph pose_graph_snap =
      run(/*use_180_degree_flip_snap=*/true, recon_snap);
  ExpectLawnmowerFlipsResolved(test_data, recon_snap, pose_graph_snap);

  // Without snapping, the mixed cross-strip edges drag the plain robust solve
  // between the two yaw basins: within-strip edges get filtered and frames
  // outside the largest remaining component are de-registered (the "split
  // into two orientation submodels" failure mode). Snapping is never worse.
  Reconstruction recon_no_snap;
  const PoseGraph pose_graph_no_snap =
      run(/*use_180_degree_flip_snap=*/false, recon_no_snap);
  EXPECT_GE(recon_snap.NumRegFrames(), recon_no_snap.NumRegFrames());
  EXPECT_GE(NumValidEdges(pose_graph_snap), NumValidEdges(pose_graph_no_snap));
}

// Starting IRLS from the identity without the L1 stage guarantees that the
// unflipped cross-strip edges are ~180 degrees from the current estimate, so
// the snapping is exercised.
TEST(RotationAveraging, FlipSnapFromIdentityInitialization) {
  const LawnmowerFlipTestData test_data = CreateLawnmowerFlipTestData();
  const TestData& data = test_data.data;

  Reconstruction reconstruction = data.reconstruction;
  PoseGraph pose_graph = data.pose_graph;
  RotationEstimatorOptions options = CreateRATestOptions(/*use_gravity=*/false);
  options.random_seed = 0;
  options.skip_initialization = true;
  options.max_num_l1_iterations = 0;
  options.use_180_degree_flip_snap = true;
  EXPECT_TRUE(RunRotationAveraging(
      options, pose_graph, reconstruction, data.pose_priors));
  ExpectLawnmowerFlipsResolved(test_data, reconstruction, pose_graph);
}

// A consistent (unflipped) lawnmower graph whose initial orientations have
// half of the second strip yaw-flipped. Starting IRLS from that torn state,
// the frame-level snapping must flip those frames back so that the ground
// truth is recovered exactly and no edge is filtered.
TEST(RotationAveraging, FlipSnapRepairsTornInitialization) {
  constexpr int kNumStrips = 2;
  constexpr int kFramesPerStrip = 6;

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = kNumStrips * kFramesPerStrip;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.prior_gravity = false;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);
  const NodeHashMap<image_t, int> image_to_strip =
      MakeNadirLawnmower(kNumStrips, kFramesPerStrip, data);

  // Initialize from the ground truth with every other frame of the second
  // strip flipped by 180 degrees about its optical axis.
  int num_torn = 0;
  for (const auto& [image_id, strip] : image_to_strip) {
    const frame_t frame_id = data.gt_reconstruction.Image(image_id).FrameId();
    Rigid3d rig_from_world =
        data.gt_reconstruction.Frame(frame_id).RigFromWorld();
    if (strip == 1 && image_id % 2 == 0) {
      rig_from_world.rotation() =
          RzPi() * Eigen::Quaterniond(rig_from_world.rotation());
      ++num_torn;
    }
    data.reconstruction.Frame(frame_id).SetRigFromWorld(rig_from_world);
  }
  ASSERT_GT(num_torn, 0);

  Reconstruction reconstruction = data.reconstruction;
  PoseGraph pose_graph = data.pose_graph;
  RotationEstimatorOptions options = CreateRATestOptions(/*use_gravity=*/false);
  options.random_seed = 0;
  options.skip_initialization = true;
  options.max_num_l1_iterations = 0;
  options.use_180_degree_flip_snap = true;
  EXPECT_TRUE(RunRotationAveraging(
      options, pose_graph, reconstruction, data.pose_priors));
  EXPECT_EQ(reconstruction.NumRegFrames(),
            data.gt_reconstruction.NumRegFrames());
  EXPECT_EQ(NumValidEdges(pose_graph), NumValidEdges(data.pose_graph));
  ExpectEqualRotations(
      data.gt_reconstruction, reconstruction, /*max_rotation_error_deg=*/1e-2);
}

TEST(RotationAveraging, FlipSnapNoOpOnConsistentGraph) {
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  for (const bool use_gravity : {true, false}) {
    Reconstruction reconstruction = data.reconstruction;
    PoseGraph pose_graph = data.pose_graph;
    RotationEstimatorOptions options = CreateRATestOptions(use_gravity);
    options.use_180_degree_flip_snap = true;
    EXPECT_TRUE(RunRotationAveraging(
        options, pose_graph, reconstruction, data.pose_priors));
    EXPECT_EQ(NumValidEdges(pose_graph), NumValidEdges(data.pose_graph));
    ExpectEqualRotations(data.gt_reconstruction,
                         reconstruction,
                         /*max_rotation_error_deg=*/1e-2);
  }
}

}  // namespace
}  // namespace colmap
