#cython: boundscheck=False, wraparound=False

from libcpp.string cimport string
from libcpp.vector cimport vector

cimport numpy as np

import numpy as np
from struct import pack, unpack, calcsize

def registerPointCloud(np.ndarray[np.float_t, ndim=2] unregisteredDepthMap,
                       np.ndarray[np.uint8_t, ndim=3] rgbImage,
                       np.ndarray[np.float_t, ndim=2] depthK,
                       np.ndarray[np.float_t, ndim=2] rgbK,
                       np.ndarray[np.float_t, ndim=2] H_RGBFromDepth):

    cdef int unregistered_height = unregisteredDepthMap.shape[0]
    cdef int unregistered_width = unregisteredDepthMap.shape[1]

    cdef int registered_height = rgbImage.shape[0]
    cdef int registered_width = rgbImage.shape[1]

    cdef np.ndarray[np.uint16_t, ndim=2] registeredDepthMap
    registeredDepthMap = np.zeros((registered_height, registered_width))

    cdef int u, v

    cdef np.float_t depth

    cdef np.ndarray[np.float_t, ndim=2] xyz_depth = np.empty((4,1))
    cdef np.ndarray[np.float_t, ndim=2] xyz_rgb = np.empty((4,1))

    # Ensure that the last value is 1 (homogeneous coordinates)
    xyz_depth [3] = 1

    cdef np.float_t invDepthFx = 1.0 / depthK[0,0]
    cdef np.float_t invDepthFy = 1.0 / depthK[1,1]
    cdef np.float_t depthCx = depthK[0,2] - depth_x_offset
    cdef np.float_t depthCy = depthK[1,2] - depth_y_offset

    cdef np.float_t rgbFx = rgbK[0,0]
    cdef np.float_t rgbFy = rgbK[1,1]
    cdef np.float_t rgbCx = rgbK[0,2]
    cdef np.float_t rgbCy = rgbK[1,2]

    cdef np.float_t invRGB_Z
    cdef int uRGB, vRGB

    cdef np.float_t registeredDepth

    for v in range(unregistered_height):
        for u in range(unregistered_width):

            depth = unregisteredDepthMap[v,u]
            if depth == 0:
                continue

            xyz_depth[0] = ((u - depthCx) * depth) * invDepthFx
            xyz_depth[1] = ((v - depthCy) * depth) * invDepthFy
            xyz_depth[2] = depth

            np.dot(H_RGBFromDepth, xyz_depth, xyz_rgb)

            invRGB_Z  = 1.0 / xyz_rgb[2]
            uRGB = (rgbFx * xyz_rgb[0]) * invRGB_Z + rgbCx + 0.5
            vRGB = (rgbFy * xyz_rgb[1]) * invRGB_Z + rgbCy + 0.5

            if (uRGB < 0 or uRGB >= registered_width) or (vRGB < 0 or vRGB >= registered_height):
                continue

            registeredDepth = xyz_rgb[2]
            if registeredDepth > registered[vRGB,uRGB]:
                registered[vRGB,uRGB] = registeredDepth

def unregisteredDepthMapToPointCloud(np.ndarray[np.float_t, ndim=2] depthMap,
                                          np.ndarray[np.float_t, ndim=2] depthK=None):

    # Use the default value that Primesense uses for most sensors if no
    # calibration matrix is provided.
    if depthK is None:
        depthK = np.array([[570.34, 0, 320.0],
                            [0, 570.34, 240.0],
                            [0, 0, 1]])
        if depthMap.shape[1] != 640:
            scale = depthMap.shape[1] / 640.0
            depthK = depthK * scale

    cdef np.float_t depthCx = depthK[0,2]
    cdef np.float_t depthCy = depthK[1,2]
    cdef np.float_t depthInvFx = 1.0/depthK[0,0]
    cdef np.float_t depthInvFy = 1.0/depthK[1,1]

    cdef int height = depthMap.shape[0]
    cdef int width = depthMap.shape[1]
    cdef int u, v

    cdef np.float_t depth

    # Allocate empty ndarray. Fill in zeros in the same pass as when creating
    # the cloud.
    cdef np.ndarray[np.float_t,ndim=3] cloud = np.empty((height, width, 3), dtype=np.float)

    for v in range(height):
        for u in range(width):
            depth = depthMap[v,u]
            if depth <= 0:
                cloud[v,u,0] = 0
                cloud[v,u,1] = 0
                cloud[v,u,2] = 0
            else:
                cloud[v,u,0] = (u - depthCx) * depth * depthInvFx
                cloud[v,u,1] = (v - depthCy) * depth * depthInvFy
                cloud[v,u,2] = depth

def registeredDepthMapToPointCloud(np.ndarray[np.float_t, ndim=2] depthMap,
                                   np.ndarray[np.uint8_t, ndim=3] rgbImage,
                                   np.ndarray[np.float_t, ndim=2] rgbK=None):

    # Use the default value that Primesense uses for most sensors if no
    # calibration matrix is provided.
    if depthK is None:
        depthK = np.array([[540, 0, 320.0],
                            [0, 540, 240.0],
                            [0, 0, 1]])
        if depthMap.shape[1] != 640:
            scale = depthMap.shape[1] / 640.0
            depthK = depthK * scale

    cdef np.float_t rgbCx = rgbK[0,2]
    cdef np.float_t rgbCy = rgbK[1,2]
    cdef np.float_t invRGBFx = 1.0/rgbK[0,0]
    cdef np.float_t invRGBFy = 1.0/rgbK[1,1]

    cdef int height = depthMap.shape[0]
    cdef int width = depthMap.shape[1]
    cdef int u, v

    for v in range(height):
        for u in range(width):
            depth = depthMap[v,u]
            if depth <= 0:
                cloud[v,u,0] = 0
                cloud[v,u,1] = 0
                cloud[v,u,2] = 0
                cloud[v,u,3] = 0
                cloud[v,u,4] = 0
                cloud[v,u,5] = 0
            else:
                cloud[v,u,0] = (u - rgbCx) * depth * invRGBFx
                cloud[v,u,1] = (v - rgbCy) * depth * invRGBFy
                cloud[v,u,2] = depth
                cloud[v,u,3] = rgbImage[v,u,0]
                cloud[v,u,4] = rgbImage[v,u,1]
                cloud[v,u,5] = rgbImage[v,u,2]

def depthMapToImage(image):
    return np.uint8(image / (np.max(image)*1.0/255))

# TODO sanity check speed / read write
def writePCD(pointCloud, filename, ascii=False):
    with open(filename, 'w') as f:
        height = pointCloud.shape[0]
        width = pointCloud.shape[1]
        f.write("# .PCD v.7 - Point Cloud Data file format\n")
        f.write("VERSION .7\n")
        if pointCloud.shape[2] == 3:
            f.write("FIELDS x y z\n")
            f.write("SIZE 4 4 4\n")
            f.write("TYPE F F F\n")
            f.write("COUNT 1 1 1\n")
        else:
            f.write("FIELDS x y z rgb\n")
            f.write("SIZE 4 4 4 4\n")
            f.write("TYPE F F F F\n")
            f.write("COUNT 1 1 1 1\n")
        f.write("WIDTH %d\n" % width)
        f.write("HEIGHT %d\n" % height)
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write("POINTS %d\n" % (height * width))
        if ascii:
          f.write("DATA ascii\n")
          for row in range(height):
            for col in range(width):
                if pointCloud.shape[2]== 3:
                    f.write("%f %f %f\n" % tuple(pointCloud[row, col, :]))
                else:
                    f.write("%f %f %f" % tuple(pointCloud[row, col, :3]))
                    r = int(pointCloud[row, col, 3])
                    g = int(pointCloud[row, col, 4])
                    b = int(pointCloud[row, col, 5])
                    rgb_int = (r << 16) | (g << 8) | b
                    packed = pack('i', rgb_int)
                    rgb = unpack('f', packed)[0]
                    f.write(" %.12e\n" % rgb)
        else:
          f.write("DATA binary\n")
          if pointCloud.shape[2] == 6:
              dt = np.dtype([('x', np.float32),
                             ('y', np.float32),
                             ('z', np.float32),
                             ('b', np.uint8),
                             ('g', np.uint8),
                             ('r', np.uint8),
                             ('I', np.uint8)])
              pointCloud_tmp = np.zeros((6, height*width, 1), dtype=dt)
              for i, k in enumerate(['x', 'y', 'z', 'r', 'g', 'b']):
                  pointCloud_tmp[k] = pointCloud[:, :, i].reshape((height*width, 1))
              pointCloud_tmp.tofile(f)
          else:
              dt = np.dtype([('x', np.float32),
                             ('y', np.float32),
                             ('z', np.float32),
                             ('I', np.uint8)])
              pointCloud_tmp = np.zeros((3, height*width, 1), dtype=dt)
              for i, k in enumerate(['x', 'y', 'z']):
                  pointCloud_tmp[k] = pointCloud[:, :, i].reshape((height*width, 1))
              pointCloud_tmp.tofile(f)

# TODO sanity check speed / read write
def readPCD(filename):
    with open(filename, 'r') as f:
        #"# .PCD v.7 - Point Cloud Data file format\n"
        f.readline()

        #"VERSION .7\n"
        f.readline()

        # "FIELDS x y z\n"
        fields = f.readline().strip().split()[1:]

        if len(fields) == 3:
            rgb = False
        elif len(fields) == 4:
            rgb = True
        else:
            raise Exception("Unsupported fields: %s" % str(fields))

        #"SIZE 4 4 4\n"
        sizes = [int(x) for x in f.readline().strip().split()[1:]]
        pointSize = np.sum(sizes)

        #"TYPE F F F\n"
        types = f.readline().strip().split()[1:]

        #"COUNT 1 1 1\n"
        counts = [int(x) for x in f.readline().strip().split()[1:]]

        #"WIDTH %d\n" % width
        width = int(f.readline().strip().split()[1])

        #"HEIGHT %d\n" % height
        height = int(f.readline().strip().split()[1])

        #"VIEWPOINT 0 0 0 1 0 0 0\n"
        viewpoint = np.array(f.readline().strip().split()[1:])

        #"POINTS %d\n" % height * width
        points = int(f.readline().strip().split()[1])

        #"DATA ascii\n"
        format = f.readline().strip().split()[1]
        ascii = format == 'ascii'

        if rgb:
            pointCloud = np.empty((height, width, 6))
        else:
            pointCloud = np.empty((height, width, 3))

        for row in range(height):
            for col in range(width):
                if ascii:
                    data = [float(x) for x in f.readline().strip().split()]
                else:
                    data = unpack('ffff', f.read(pointSize))

                pointCloud[row, col, 0] = data[0]
                pointCloud[row, col, 1] = data[1]
                pointCloud[row, col, 2] = data[2]
                if rgb:
                    rgb_float = data[3]
                    packed = pack('f', rgb_float)
                    rgb_int = unpack('i', packed)[0]
                    r = rgb_int >> 16 & 0x0000ff
                    g = rgb_int >> 8 & 0x0000ff
                    b = rgb_int & 0x0000ff
                    pointCloud[row, col, 3] = r
                    pointCloud[row, col, 4] = g
                    pointCloud[row, col, 5] = b

        return pointCloud
