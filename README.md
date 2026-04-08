# WebXR Pointcloud Viewer

Simple WebXR/Three.js viewer for `.ply` and `.pcd` point clouds, plus a helper script for correcting the orientation and origin of clouds extracted from MCAPs.

## Pointcloud Correction

Use `pointcloud_transform.py` to rewrite a point cloud with a corrected origin, rotation, and optional translation.

Example command:

```bash
python pointcloud_transform.py "C:\Users\logun\Desktop\WebXR_Pointcloud_Viewer\Pointclouds\colored_cloud.ply" "C:\Users\logun\Desktop\WebXR_Pointcloud_Viewer\Pointclouds\colored_cloud_transformed.ply" --origin -38.783 -171.17 25.8399 --rotate -86 0 0 --translate -41.5876 13.6943 169.55
```

Values used for this correction:

- `origin`: `-38.783 -171.17 25.8399`
- `rotate`: `-86 0 0`
- `translate`: `-41.5876 13.6943 169.55`
