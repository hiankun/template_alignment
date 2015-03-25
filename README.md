# Note

* This is just for personal test.
* The code was modified from [Aligning object templates to a point cloud](http://www.pointclouds.org/documentation/tutorials/template_alignment.php).

# Usage

Run the following command:

`./template_alignment ../data/object_templates.txt ../data/person.pcd`

And if you have PCL installed in your system (which is Ubuntu 14.04 in my case), you might be able to view the alignment result by using:

`/usr/bin/pcl_viewer output.pcd ../data/<your_scene_file>.pcd `

