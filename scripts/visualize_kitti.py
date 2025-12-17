"""
Visualize KITTI point clouds and 3D bounding boxes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
import argparse
from pathlib import Path


def load_velodyne(filepath):
    """Load point cloud from .bin file"""
    points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
    return points


def load_labels(filepath):
    """Load labels from .txt file"""
    if not Path(filepath).exists():
        return []
    
    labels = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(' ')
            if parts[0] in ['Car', 'Pedestrian', 'Cyclist']:
                label = {
                    'type': parts[0],
                    'dimensions': [float(x) for x in parts[8:11]],  # h, w, l
                    'location': [float(x) for x in parts[11:14]],   # x, y, z
                    'rotation_y': float(parts[14])
                }
                labels.append(label)
    return labels


def plot_bev(points, labels=None, save_path=None):
    """Plot bird's eye view of point cloud with boxes"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Plot points (subsample for speed)
    subsample = points[::5]  # Every 5th point
    ax.scatter(subsample[:, 0], subsample[:, 1], 
               s=0.5, c=subsample[:, 2], cmap='viridis', alpha=0.5)
    
    # Plot bounding boxes
    if labels:
        colors = {'Car': 'red', 'Pedestrian': 'blue', 'Cyclist': 'green'}
        
        for label in labels:
            obj_type = label['type']
            h, w, l = label['dimensions']
            x, y, z = label['location']
            ry = label['rotation_y']
            
            # Create rectangle (simplified - camera coords)
            rect = Rectangle(
                (x - l/2, y - w/2), l, w,
                linewidth=2,
                edgecolor=colors.get(obj_type, 'yellow'),
                facecolor='none',
                label=obj_type
            )
            
            # Apply rotation
            t = Affine2D().rotate(ry).translate(x, y) + ax.transData
            rect.set_transform(t)
            ax.add_patch(rect)
    
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Bird\'s Eye View - LiDAR')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Legend
    if labels:
        handles = [plt.Line2D([0], [0], color=c, linewidth=2, label=t) 
                   for t, c in colors.items()]
        ax.legend(handles=handles, loc='upper right')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_3d(points, labels=None):
    """Plot 3D visualization"""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Subsample points
    subsample = points[::10]
    
    # Color by height
    ax.scatter(subsample[:, 0], subsample[:, 1], subsample[:, 2],
               c=subsample[:, 2], cmap='viridis', s=0.1, alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_zlim(-3, 5)
    
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--velodyne', type=str, required=True,
                        help='Path to .bin velodyne file')
    parser.add_argument('--label', type=str, default=None,
                        help='Path to .txt label file')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save visualization')
    parser.add_argument('--view', type=str, default='bev',
                        choices=['bev', '3d', 'both'],
                        help='Visualization type')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading {args.velodyne}...")
    points = load_velodyne(args.velodyne)
    print(f"Loaded {len(points)} points")
    
    labels = None
    if args.label:
        labels = load_labels(args.label)
        print(f"Loaded {len(labels)} labels")
    
    # Visualize
    if args.view in ['bev', 'both']:
        plot_bev(points, labels, args.save)
    
    if args.view in ['3d', 'both']:
        plot_3d(points, labels)


if __name__ == '__main__':
    main()