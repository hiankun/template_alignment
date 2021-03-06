#include <limits>
#include <fstream>
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/pfh.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/console/parse.h>
#include <pcl/visualization/point_cloud_handlers.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/susan.h>
#include <pcl/keypoints/uniform_sampling.h>

#include <pcl/range_image/range_image_planar.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/features/narf_descriptor.h>
#include <pcl/visualization/range_image_visualizer.h>

enum KP_METHOD {VOX, SIFT, ISS, SUSAN, UNI, NARF};
typedef pcl::PointXYZRGBA PointType;

class FeatureCloud
{
    public:
        // A bit of shorthand
        typedef pcl::PointCloud<PointType> PointCloud;
        typedef pcl::PointCloud<pcl::Normal> SurfaceNormals;
        typedef pcl::PointCloud<pcl::FPFHSignature33> LocalFeatures;
        typedef pcl::search::KdTree<PointType> SearchMethod;

        FeatureCloud () :
            search_method_xyz_ (new SearchMethod),
            normal_radius_ (0.02f),
            feature_radius_ (0.02f)
    {}

        ~FeatureCloud () {}

        // Process the given cloud
        void
            setInputCloud (PointCloud::Ptr xyz)
            {
                xyz_ = xyz;
                processInput ();
            }

        // Load and process the cloud in the given PCD file
        void
            loadInputCloud (const std::string &pcd_file)
            {
                xyz_ = PointCloud::Ptr (new PointCloud);
                pcl::io::loadPCDFile (pcd_file, *xyz_);
                processInput ();
            }

        // Get a pointer to the cloud 3D points
        PointCloud::Ptr
            getPointCloud () const
            {
                return (xyz_);
            }

        // Get a pointer to the cloud of 3D surface normals
        SurfaceNormals::Ptr
            getSurfaceNormals () const
            {
                return (normals_);
            }

        // Get a pointer to the cloud of feature descriptors
        LocalFeatures::Ptr
            getLocalFeatures () const
            {
                return (features_);
            }

    protected:
        // Compute the surface normals and local features
        void
            processInput ()
            {
                computeSurfaceNormals ();
                computeLocalFeatures ();
            }

        // Compute the surface normals
        void
            computeSurfaceNormals ()
            {
                normals_ = SurfaceNormals::Ptr (new SurfaceNormals);

                pcl::NormalEstimation<PointType, pcl::Normal> norm_est;
                norm_est.setInputCloud (xyz_);
                norm_est.setSearchMethod (search_method_xyz_);
                norm_est.setRadiusSearch (normal_radius_);
                norm_est.compute (*normals_);
            }

        // Compute the local feature descriptors
        void
            computeLocalFeatures ()
            {
                features_ = LocalFeatures::Ptr (new LocalFeatures);

                pcl::FPFHEstimation<PointType, pcl::Normal, pcl::FPFHSignature33> fpfh_est;
                fpfh_est.setInputCloud (xyz_);
                fpfh_est.setInputNormals (normals_);
                fpfh_est.setSearchMethod (search_method_xyz_);
                fpfh_est.setRadiusSearch (feature_radius_);
                fpfh_est.compute (*features_);
            }

    private:
        // Point cloud data
        PointCloud::Ptr xyz_;
        SurfaceNormals::Ptr normals_;
        LocalFeatures::Ptr features_;
        SearchMethod::Ptr search_method_xyz_;

        // Parameters
        float normal_radius_;
        float feature_radius_;
        int feature_method_;
};

class TemplateAlignment
{
    public:

        // A struct for storing alignment results
        struct Result
        {
            float fitness_score;
            Eigen::Matrix4f final_transformation;
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        };

        TemplateAlignment () :
            min_sample_distance_ (0.05f),
            max_correspondence_distance_ (0.01f*0.01f),
            nr_iterations_ (500)
    {
        // Intialize the parameters in the Sample Consensus Intial Alignment (SAC-IA) algorithm
        sac_ia_.setMinSampleDistance (min_sample_distance_);
        sac_ia_.setMaxCorrespondenceDistance (max_correspondence_distance_);
        sac_ia_.setMaximumIterations (nr_iterations_);
    }

        ~TemplateAlignment () {}

        // Set the given cloud as the target to which the templates will be aligned
        void
            setTargetCloud (FeatureCloud &target_cloud)
            {
                target_ = target_cloud;
                sac_ia_.setInputTarget (target_cloud.getPointCloud ());
                sac_ia_.setTargetFeatures (target_cloud.getLocalFeatures ());
            }

        // Add the given cloud to the list of template clouds
        void
            addTemplateCloud (FeatureCloud &template_cloud)
            {
                templates_.push_back (template_cloud);
            }

        // Align the given template cloud to the target specified by setTargetCloud ()
        void
            align (FeatureCloud &template_cloud, TemplateAlignment::Result &result)
            {
                //sac_ia_.setInputCloud (template_cloud.getPointCloud ());
                sac_ia_.setInputSource (template_cloud.getPointCloud ());
                sac_ia_.setSourceFeatures (template_cloud.getLocalFeatures ());

                pcl::PointCloud<PointType> registration_output;
                sac_ia_.align (registration_output);

                result.fitness_score = (float) sac_ia_.getFitnessScore (max_correspondence_distance_);
                result.final_transformation = sac_ia_.getFinalTransformation ();
            }

        // Align all of template clouds set by addTemplateCloud to the target specified by setTargetCloud ()
        void
            alignAll (std::vector<TemplateAlignment::Result, Eigen::aligned_allocator<Result> > &results)
            {
                results.resize (templates_.size ());
                for (size_t i = 0; i < templates_.size (); ++i)
                {
                    align (templates_[i], results[i]);
                }
            }

        // Align all of template clouds to the target cloud to find the one with best alignment score
        int
            findBestAlignment (TemplateAlignment::Result &result)
            {
                // Align all of the templates to the target cloud
                std::vector<Result, Eigen::aligned_allocator<Result> > results;
                alignAll (results);

                // Find the template with the best (lowest) fitness score
                float lowest_score = std::numeric_limits<float>::infinity ();
                int best_template = 0;
                for (size_t i = 0; i < results.size (); ++i)
                {
                    const Result &r = results[i];
                    if (r.fitness_score < lowest_score)
                    {
                        lowest_score = r.fitness_score;
                        best_template = (int) i;
                    }
                }

                // Output the best alignment
                result = results[best_template];
                return (best_template);
            }

    private:
        // A list of template clouds and the target to which they will be aligned
        std::vector<FeatureCloud> templates_;
        FeatureCloud target_;

        // The Sample Consensus Initial Alignment (SAC-IA) registration routine and its parameters
        pcl::SampleConsensusInitialAlignment<PointType, PointType, pcl::FPFHSignature33> sac_ia_;
        float min_sample_distance_;
        float max_correspondence_distance_;
        int nr_iterations_;
};

// This function by Tommaso Cavallari and Federico Tombari, taken from the tutorial
// http://pointclouds.org/documentation/tutorials/correspondence_grouping.php
double computeCloudResolution(const pcl::PointCloud<PointType>::ConstPtr& cloud)
{
    double resolution = 0.0;
    int numberOfPoints = 0;
    int nres;
    std::vector<int> indices(2);
    std::vector<float> squaredDistances(2);
    pcl::search::KdTree<PointType> tree;
    tree.setInputCloud(cloud);

    for (size_t i = 0; i < cloud->size(); ++i)
    {
        if (! pcl_isfinite((*cloud)[i].x))
            continue;

        // Considering the second neighbor since the first is the point itself.
        nres = tree.nearestKSearch(i, 2, indices, squaredDistances);
        if (nres == 2)
        {
            resolution += sqrt(squaredDistances[1]);
            ++numberOfPoints;
        }
    }
    if (numberOfPoints != 0)
        resolution /= numberOfPoints;

    return resolution;
}

void get_range_image(pcl::PointCloud<PointType>::Ptr &cloud,
      pcl::RangeImagePlanar& range_image) {
    float image_size_x = 640; //cloud->width;
    float image_size_y = 480; //cloud->height;

    float center_x = image_size_x * 0.5f;
    float center_y = image_size_y * 0.5f;
    float focal_length_x = 200.0f; //todo
    float focal_length_y = focal_length_x;

    Eigen::Affine3f scene_sensor_pose = Eigen::Affine3f(Eigen::Translation3f(
                cloud->sensor_origin_[0],
                cloud->sensor_origin_[1],
                cloud->sensor_origin_[2])) *
        Eigen::Affine3f(cloud->sensor_orientation_);

    pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
    float noise_level = 0.0f;
    float min_range = 0.0f;

    //pcl::RangeImagePlanar range_image;
    range_image.createFromPointCloudWithFixedSize(
            *cloud, image_size_x, image_size_y,
            center_x, center_y, focal_length_x, focal_length_y,
            scene_sensor_pose, coordinate_frame,
            noise_level, min_range);
#if 0
    //-- visualization
    pcl::visualization::RangeImageVisualizer viewer("planar range image");
    viewer.showRangeImage(range_image);
    while (!viewer.wasStopped()) {
        viewer.spinOnce();
        pcl_sleep(0.1);
    }
#endif
}

// Align a collection of object templates to a sample point cloud
    int
main (int argc, char **argv)
{
    if (argc < 3)
    {
        //printf ("No target PCD file given!\n");
        std::cout << "usage: ./template_alignment ../data/object_templates.txt ../data/person.pcd [-vox|-sift|-iss|-uni|-narf]\n";
        return (-1);
    }

    // Load the object templates specified in the object_templates.txt file
    std::vector<FeatureCloud> object_templates;
    std::ifstream input_stream (argv[1]);
    object_templates.resize (0);
    std::string pcd_filename;
    while (input_stream.good ())
    {
        std::getline (input_stream, pcd_filename);
        if (pcd_filename.empty () || pcd_filename.at (0) == '#') // Skip blank lines or comments
            continue;

        FeatureCloud template_cloud;
        template_cloud.loadInputCloud (pcd_filename);
        object_templates.push_back (template_cloud);
    }
    input_stream.close ();

    // Load the target cloud PCD file
    //pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType>);
    pcl::io::loadPCDFile (argv[2], *cloud);

#if 0 //-- the filter cut out part of the cloud
    // Preprocess the cloud by...
    // ...removing distant points
    const float depth_limit = 1.0;
    //pcl::PassThrough<PointType> pass;
    pcl::PassThrough<PointType> pass;
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0, depth_limit);
    pass.filter (*cloud);
#endif

    KP_METHOD kp_method;
    //int method = 0;

    float voxel_grid_size = 0.005f;
    if (pcl::console::parse(argc, argv, "-vox", voxel_grid_size) > 0)
        kp_method = VOX;

    float min_scale = 0.01f;
    if (pcl::console::parse(argc, argv, "-sift", min_scale) > 0)
        kp_method = SIFT;

    double cloud_resolution (0.0058329);
    if (pcl::console::parse(argc, argv, "-iss", cloud_resolution) > 0)
        kp_method = ISS;

    float susan_radius (0.01f);
    if (pcl::console::parse(argc, argv, "-susan", susan_radius) > 0)
        kp_method = SUSAN;

    float uni_radius (0.01f);
    if (pcl::console::parse(argc, argv, "-uni", uni_radius) > 0)
        kp_method = UNI;

    float support_size = 0.2f; //todo
    if (pcl::console::parse(argc, argv, "-narf", support_size) > 0)
        kp_method = NARF;

    FeatureCloud target_cloud;
    pcl::PointCloud<PointType>::Ptr cloud_kp (new pcl::PointCloud<PointType>);
    switch (kp_method)
    {
        case VOX:
            {
                // ... just downsample the point cloud
                pcl::VoxelGrid<PointType> vox_grid;
                vox_grid.setInputCloud (cloud);
                vox_grid.setLeafSize (voxel_grid_size, voxel_grid_size, voxel_grid_size);
                //vox_grid.filter (*cloud); // Please see this http://www.pcl-developers.org/Possible-problem-in-new-VoxelGrid-implementation-from-PCL-1-5-0-td5490361.html
                //pcl::PointCloud<PointType>::Ptr tempCloud (new pcl::PointCloud<PointType>);
                pcl::PointCloud<PointType>::Ptr tempCloud (new pcl::PointCloud<PointType>);
                vox_grid.filter (*tempCloud);
                std::cout << "Downsampled points are of size: " << tempCloud->points.size () <<std::endl;

                //-- workaround, to convert the cloud from XYZRGB to XYZ
                //pcl::PointCloud<PointType>::Ptr cloud_xyz (new pcl::PointCloud<PointType>);
                //pcl::copyPointCloud(*tempCloud, *cloud_xyz);

                // Assign to the target FeatureCloud
                target_cloud.setInputCloud (tempCloud);
                break;
            }
        case SIFT:
            {
                //-- keypoints
                // Parameters for sift computation
                const int n_octaves = 6;
                const int n_scales_per_octave = 10;
                const float min_contrast = 0.5f;
                // Estimate the sift interest points using Intensity values from RGB values
                pcl::SIFTKeypoint<PointType, pcl::PointWithScale> sift;
                pcl::PointCloud<pcl::PointWithScale> result;
                pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType> ());
                sift.setSearchMethod(tree);
                sift.setScales(min_scale, n_octaves, n_scales_per_octave);
                sift.setMinimumContrast(min_contrast);
                sift.setInputCloud(cloud);
                sift.compute(result);
                pcl::copyPointCloud(result, *cloud_kp);
                // Saving the resultant cloud
                std::cout << "Resulting Sift 3D points are of size: " << cloud_kp->points.size () <<std::endl;
                //pcl::io::savePCDFileASCII("sift_3d_kp.pcd", *cloud_kp);
                //
                // Assign to the target FeatureCloud
                target_cloud.setInputCloud (cloud_kp);
                break;
            }
        case ISS:
            {
                pcl::PointCloud<PointType>::Ptr keypoints(new pcl::PointCloud<PointType>);
                // ISS keypoint detector object.
                pcl::ISSKeypoint3D<PointType, PointType> iss;
                //iss.setInputCloud(cloud_xyz);
                iss.setInputCloud(cloud);
                //pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>);
                //iss.setSearchMethod(kdtree);

                if (cloud_resolution <= 0.0)
                    cloud_resolution = computeCloudResolution(cloud);
                std::cout << "cloud_resolution: " << cloud_resolution << std::endl;

                // Set the radius of the spherical neighborhood used to compute the scatter matrix.
                iss.setSalientRadius(6 * cloud_resolution);
                // Set the radius for the application of the non maxima supression algorithm.
                iss.setNonMaxRadius(4 * cloud_resolution);
                // Set the minimum number of neighbors that has to be found while applying the non maxima suppression algorithm.
                iss.setMinNeighbors(5);
                // Set the upper bound on the ratio between the second and the first eigenvalue.
                iss.setThreshold21(0.975);
                // Set the upper bound on the ratio between the third and the second eigenvalue.
                iss.setThreshold32(0.975);
                // Set the number of prpcessing threads to use. 0 sets it to automatic.
                iss.setNumberOfThreads(0);

                iss.compute(*keypoints);
                pcl::copyPointCloud(*keypoints, *cloud_kp);
                std::cout << "Resulting ISS 3D points are of size: " << cloud_kp->points.size () <<std::endl;
                //pcl::io::savePCDFileASCII("iss_3d_kp.pcd", *cloud_kp);

                // Assign to the target FeatureCloud
                target_cloud.setInputCloud (cloud_kp);

                break;
            }
        case SUSAN:
            {
                pcl::PointCloud<PointType>::Ptr keypoints(new pcl::PointCloud<PointType>);
                // SUSAN keypoint detector object.
                pcl::SUSANKeypoint<PointType, PointType> susan;

                susan.setInputCloud(cloud);

                susan.setRadius(susan_radius);
                //susan.setDistanceThreshold(0.001);
                //susan.setAngularThreshold(0.0001);
                //susan.setIntensityThreshold(7.0);

                //susan.setSearchSurface(cloud_xyz);
                susan.setNonMaxSupression(true);
                susan.compute(*keypoints);
                //
                pcl::copyPointCloud(*keypoints, *cloud_kp);
                std::cout << "Resulting SUSAN points are of size: " << cloud_kp->points.size () <<std::endl;
                //pcl::io::savePCDFileASCII("susan_kp.pcd", *cloud_kp);
                //
                // Assign to the target FeatureCloud
                target_cloud.setInputCloud (cloud_kp);

                break;
            }
        case UNI:
            {
                pcl::UniformSampling<PointType> uni;
                uni.setRadiusSearch(uni_radius);
                uni.setInputCloud(cloud);

                pcl::PointCloud<int> keypoints_idx;
                uni.compute(keypoints_idx);

                pcl::copyPointCloud(*cloud, keypoints_idx.points, *cloud_kp);
                std::cout << "Resulting UniformSampling points are of size: " << cloud_kp->points.size () <<std::endl;

                // Assign to the target FeatureCloud
                target_cloud.setInputCloud (cloud_kp);

                break;
            }
        case NARF:
            {
                //-- get range image
                pcl::RangeImagePlanar range_image;
                get_range_image(cloud, range_image);

                //-- find borders
                pcl::RangeImageBorderExtractor border_extractor(&range_image);
                pcl::PointCloud<pcl::BorderDescription>::Ptr borders(\
                        new pcl::PointCloud<pcl::BorderDescription>);
                border_extractor.compute(*borders);

                //-- get keypoints
                pcl::NarfKeypoint detector(&border_extractor);
                detector.setRangeImage(&range_image);
                detector.getParameters().support_size = support_size;//todo
                pcl::PointCloud<int>::Ptr keypoints_idx(new pcl::PointCloud<int>);
                detector.compute(*keypoints_idx);

                pcl::copyPointCloud(*cloud, keypoints_idx->points, *cloud_kp);
                std::cout << "Resulting NARF points are of size: " << keypoints_idx->points.size () <<std::endl;
                //pcl::io::savePCDFileASCII("narf_kp.pcd", *cloud_kp);
                //
                // Assign to the target FeatureCloud
                target_cloud.setInputCloud (cloud_kp);

                break;
            }
    }


    // Set the TemplateAlignment inputs
    TemplateAlignment template_align;
    for (size_t i = 0; i < object_templates.size (); ++i)
    {
        template_align.addTemplateCloud (object_templates[i]);
    }
    template_align.setTargetCloud (target_cloud);

    // Find the best template alignment
    TemplateAlignment::Result best_alignment;
    int best_index = template_align.findBestAlignment (best_alignment);
    const FeatureCloud &best_template = object_templates[best_index];

    // Print the alignment fitness score (values less than 0.00002 are good)
    printf ("Best fitness score: %f\n", best_alignment.fitness_score);

    // Print the rotation matrix and translation vector
    Eigen::Matrix3f rotation = best_alignment.final_transformation.block<3,3>(0, 0);
    Eigen::Vector3f translation = best_alignment.final_transformation.block<3,1>(0, 3);

    printf ("\n");
    printf ("    | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
    printf ("R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
    printf ("    | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
    printf ("\n");
    printf ("t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));

    // Save the aligned template for visualization
    pcl::PointCloud<PointType>::Ptr transformed_cloud (new pcl::PointCloud<PointType>);
    pcl::transformPointCloud (*best_template.getPointCloud (), *transformed_cloud, best_alignment.final_transformation);
    pcl::io::savePCDFileBinary ("output.pcd", *transformed_cloud);


    //-- visualization
    //-- the scene
    pcl::visualization::PCLVisualizer viewer(argv[1]);
    viewer.addPointCloud(cloud, "original");

    //-- the keypoints (or downsampled points)
    pcl::visualization::PointCloudColorHandlerCustom<PointType> kp_colorHandler(cloud_kp, 255, 255, 0);
    viewer.addPointCloud(cloud_kp,kp_colorHandler, "kp");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "kp");
    //-- the transformed template
    pcl::visualization::PointCloudColorHandlerCustom<PointType> tr_colorHandler(transformed_cloud, 255, 0, 0);
    viewer.addPointCloud(transformed_cloud, tr_colorHandler, "transformed");

    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }

    return (0);
}
