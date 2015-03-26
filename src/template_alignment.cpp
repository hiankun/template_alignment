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

//#define USE_KP 1

class FeatureCloud
{
    public:
        // A bit of shorthand
        typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
        typedef pcl::PointCloud<pcl::Normal> SurfaceNormals;
        typedef pcl::PointCloud<pcl::FPFHSignature33> LocalFeatures;
        typedef pcl::search::KdTree<pcl::PointXYZ> SearchMethod;

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

                pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> norm_est;
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

                pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_est;
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

                pcl::PointCloud<pcl::PointXYZ> registration_output;
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
        pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia_;
        float min_sample_distance_;
        float max_correspondence_distance_;
        int nr_iterations_;
};

// Align a collection of object templates to a sample point cloud
    int
main (int argc, char **argv)
{
    if (argc < 3)
    {
        //printf ("No target PCD file given!\n");
        std::cout << "usage: ./template_alignment ../data/object_templates.txt ../data/person.pcd [-s <sift3D min_scale> | -v <voxel_grid_size>]\n";
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
    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPCDFile (argv[2], *cloud);

    // Preprocess the cloud by...
    // ...removing distant points
    const float depth_limit = 1.0;
    //pcl::PassThrough<pcl::PointXYZ> pass;
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0, depth_limit);
    pass.filter (*cloud);


    bool USE_KP = false;

    float min_scale = 0.01f;
    if (pcl::console::parse(argc, argv, "-s", min_scale) > 0)
        USE_KP = true;

    float voxel_grid_size = 0.005f;
    if (pcl::console::parse(argc, argv, "-v", voxel_grid_size) > 0)
        USE_KP = false;

    FeatureCloud target_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr kp_view (new pcl::PointCloud<pcl::PointXYZ>);
    if (USE_KP) {
        //-- keypoints
        // Parameters for sift computation
        const int n_octaves = 6;
        const int n_scales_per_octave = 10;
        const float min_contrast = 0.5f;
        // Estimate the sift interest points using Intensity values from RGB values
        pcl::SIFTKeypoint<pcl::PointXYZRGB, pcl::PointWithScale> sift;
        pcl::PointCloud<pcl::PointWithScale> result;
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB> ());
        sift.setSearchMethod(tree);
        sift.setScales(min_scale, n_octaves, n_scales_per_octave);
        sift.setMinimumContrast(min_contrast);
        sift.setInputCloud(cloud);
        sift.compute(result);
        // Copying the pointwithscale to pointxyz so as visualize the cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_kp (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::copyPointCloud(result, *cloud_kp);
        // Saving the resultant cloud
        std::cout << "Resulting sift points are of size: " << cloud_kp->points.size () <<std::endl;
        pcl::io::savePCDFileASCII("sift_points.pcd", *cloud_kp);
        //
        // Assign to the target FeatureCloud
        target_cloud.setInputCloud (cloud_kp);
        //-- copy for viewer
        pcl::copyPointCloud(*cloud_kp, *kp_view);
    } else {
        // ... just downsample the point cloud
        //pcl::VoxelGrid<pcl::PointXYZ> vox_grid;
        pcl::VoxelGrid<pcl::PointXYZRGB> vox_grid;
        vox_grid.setInputCloud (cloud);
        vox_grid.setLeafSize (voxel_grid_size, voxel_grid_size, voxel_grid_size);
        //vox_grid.filter (*cloud); // Please see this http://www.pcl-developers.org/Possible-problem-in-new-VoxelGrid-implementation-from-PCL-1-5-0-td5490361.html
        //pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr tempCloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        vox_grid.filter (*tempCloud);
        std::cout << "Downsampled points are of size: " << tempCloud->points.size () <<std::endl;
        //cloud = tempCloud;
        //-- workaround, to convert the cloud from XYZRGB to XYZ
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz (new pcl::PointCloud<pcl::PointXYZ>);
        //pcl::copyPointCloud(*cloud, *cloud_xyz);
        pcl::copyPointCloud(*tempCloud, *cloud_xyz);

        // Assign to the target FeatureCloud
        target_cloud.setInputCloud (cloud_xyz);
        //-- copy for viewer
        pcl::copyPointCloud(*cloud_xyz, *kp_view);
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
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud (*best_template.getPointCloud (), *transformed_cloud, best_alignment.final_transformation);
    pcl::io::savePCDFileBinary ("output.pcd", *transformed_cloud);


    //-- visualization
    //-- the scene
    pcl::visualization::PCLVisualizer viewer(argv[1]);
    viewer.addPointCloud(cloud, "original");
    //-- the keypoints (or downsampled points)
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> kp_colorHandler(kp_view, 255, 255, 0);
    viewer.addPointCloud(kp_view,kp_colorHandler, "kp");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "kp");
    //-- the transformed template
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> tr_colorHandler(transformed_cloud, 255, 0, 0);
    viewer.addPointCloud(transformed_cloud, tr_colorHandler, "transformed");

    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }

    return (0);
}
