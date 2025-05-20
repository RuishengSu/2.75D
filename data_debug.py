
import math
import random
import numpy as np


def get_slices_from_blob(blob, blob_size):
    number_of_planes = 9
    extracted_planes = np.zeros((blob_size, blob_size, number_of_planes), dtype=blob.dtype)

    # plane 1
    extracted_planes[:, :, 0] = (blob[:, :, blob_size // 2] + blob[:, :, blob_size // 2 + 1]) / 2
    # plane 2
    extracted_planes[:, :, 1] = (blob[:, blob_size // 2, :] + blob[:, blob_size // 2 + 1, :]) / 2
    # plane 3
    extracted_planes[:, :, 2] = (blob[blob_size // 2, :, :] + blob[blob_size // 2 + 1, :, :]) / 2

    # plane 4
    extracted_planes[:, :, 3] = np.diagonal(blob[:, :, :], 0, 0, 1)
    # plane 5
    extracted_planes[:, :, 4] = np.diagonal(blob[:, :, :], 0, 0, 2)
    # plane 6
    extracted_planes[:, :, 5] = np.diagonal(blob[:, :, :], 0, 1, 2)

    # plane 7
    extracted_planes[:, :, 6] = np.diagonal(np.rot90(blob[:, :, :], k=1, axes=(0, 1)), 0, 0, 1)
    # plane 8
    extracted_planes[:, :, 7] = np.diagonal(np.rot90(blob[:, :, :], k=1, axes=(0, 2)), 0, 0, 2)
    # plane 9
    extracted_planes[:, :, 8] = np.diagonal(np.rot90(blob[:, :, :], k=1, axes=(1, 2)), 0, 1, 2)

    return extracted_planes


def convert_to_euclidean_space(blob_radius, inner_radius, alpha, beta):
    # Z is the height; space center is the top left corner of the cubic
    verticle_angle = (math.pi / 2.0) - beta
    inner_radius_XY = inner_radius * math.cos(verticle_angle)
    Z = int(round(blob_radius - inner_radius * math.sin(verticle_angle)))
    X = int(round(blob_radius + inner_radius_XY * math.cos(alpha)))
    Y = int(round(blob_radius + inner_radius_XY * math.sin(alpha)))
    return X, Y, Z


def get_angels(N):
    sphere_step_size = math.pi / float(N)
    # nr_of_points_on_each_circle = 2*N
    alpha_list = []  # horizontal angles, 0-2pi
    beta_list = []  # vitical angles, 0-pi
    alpha_list.append(0)
    beta_list.append(0)

    vertical_beta_step = math.pi / N
    for i in range(0, int(math.ceil(N / 2.0))):
        vertical_beta_on_Ith_circle = (i + 1) * vertical_beta_step
        cos_angel = math.fabs(math.pi / 2 - vertical_beta_on_Ith_circle)
        perimeter = 2 * math.pi * math.cos(cos_angel)
        nr_of_points_on_circle = int(round(perimeter / sphere_step_size))
        for j in range(0, nr_of_points_on_circle):
            alpha_step_size = 2 * math.pi / float(nr_of_points_on_circle)
            beta_step_size_between_circles = vertical_beta_step / float(nr_of_points_on_circle)
            alpha_list.append((j + 1) * alpha_step_size)
            beta_list.append(i * vertical_beta_step + (j + 1) * beta_step_size_between_circles)
    for i in range(int(math.ceil(N / 2.0)), N):
        vertical_beta_on_Ith_circle = i * vertical_beta_step
        cos_angel = math.fabs(math.pi / 2 - vertical_beta_on_Ith_circle)
        perimeter = 2 * math.pi * math.cos(cos_angel)
        nr_of_points_on_circle = int(round(perimeter / sphere_step_size))
        for j in range(0, nr_of_points_on_circle):
            alpha_step_size = 2 * math.pi / float(nr_of_points_on_circle)
            beta_step_size_between_circles = vertical_beta_step / float(nr_of_points_on_circle)
            alpha_list.append((j + 1) * alpha_step_size)
            beta_list.append(i * vertical_beta_step + (j + 1) * beta_step_size_between_circles)
    return alpha_list, beta_list


def get_spiral_image(blob, spiral_image_height):
    blob_size = blob.shape

    if blob_size[0] != blob_size[1] or blob_size[0] != blob_size[2]:
        raise ValueError("input blob is not a cubic!!")

    # Input parameters
    blob_radius = (blob_size[0] - 1) / 2.0
    nr_of_points_on_radius = spiral_image_height  # including the point on the out most sphere 32
    N = 9  # Divide a half circle pi into N parts, meaning the number of horizontal circles, including the other polar point

    alpha_list, beta_list = get_angels(N)
    radius_list = []
    radius_step = blob_radius / float(
        nr_of_points_on_radius)  # number of points on radius should be smaller than radius, since the points will be integer values
    for i in range(nr_of_points_on_radius, 0, -1):
        radius_list.append(i * radius_step)

    spiral_image = np.empty([radius_list.__len__(), alpha_list.__len__(), 1])

    for row_index in range(0, radius_list.__len__()):
        radius_for_the_row = radius_list[row_index]
        for column_index in range(0, alpha_list.__len__()):
            current_alpha = alpha_list[column_index]
            current_beta = beta_list[column_index]
            X, Y, Z = convert_to_euclidean_space(blob_radius, radius_for_the_row, current_alpha, current_beta)

            spiral_image[row_index][column_index][0] = blob[X][Y][Z]

    return spiral_image


def get_spiral_image_3channels(blob, spiral_image_height):
    blob_size = blob.shape

    if blob_size[0] != blob_size[1] or blob_size[0] != blob_size[2]:
        raise ValueError("input blob is not a cubic!!")

    # Input parameters
    blob_radius = (blob_size[0] - 1) / 2.0
    nr_of_points_on_radius = spiral_image_height  # including the point on the out most sphere
    N = 9  # Divide a half circle pi into N parts, meaning the number of horizontal circles, including the other polar point

    alpha_list, beta_list = get_angels(N)
    radius_list = []
    radius_step = blob_radius / float(
        nr_of_points_on_radius)  # number of points on radius should be smaller than radius, since the points will be integer values
    for i in range(nr_of_points_on_radius, 0, -1):
        radius_list.append(i * radius_step)

    spiral_image_3channels = np.empty([radius_list.__len__(), alpha_list.__len__(), 3])

    for row_index in range(0, radius_list.__len__()):
        radius_for_the_row = radius_list[row_index]
        for column_index in range(0, alpha_list.__len__()):
            current_alpha = alpha_list[column_index]
            current_beta = beta_list[column_index]
            X, Y, Z = convert_to_euclidean_space(blob_radius, radius_for_the_row, current_alpha, current_beta)

            spiral_image_3channels[row_index][column_index][0] = blob[X][Y][Z]
            spiral_image_3channels[row_index][column_index][1] = blob[Y][Z][X]
            spiral_image_3channels[row_index][column_index][2] = blob[Z][X][Y]

    return spiral_image_3channels


if __name__ == '__main__':
    # Create a random 64x64x64 blob
    cube_size = 64
    spiral_image_height = 32
    cube = np.random.rand(cube_size, cube_size, cube_size)

    ################################################################
    # 2D method
    img_2d = get_slices_from_blob(cube, cube_size)[:, :, 0]
    # Print the shapes of the extracted images to verify
    print(img_2d.shape)  # Should be (64, 64)
    ################################################################
    # 2.5D method
    img_25d = get_slices_from_blob(cube, cube_size)
    # Print the shapes of the extracted images to verify
    print(img_25d.shape)  # Should be (64, 64, 9)
    ################################################################
    # 2.75D method
    img_275d = get_spiral_image(cube, spiral_image_height).squeeze(2)
    # Print the shapes of the extracted images to verify
    print(img_275d.shape)  # Should be (32, 123)
    ################################################################
    # 2.75D 3channel method
    img_25d3channel = get_spiral_image_3channels(cube, spiral_image_height)
    # Print the shapes of the extracted images to verify
    print(img_25d3channel.shape)  # Should be (32, 123, 3)
    ################################################################
    # 3D method
    img_3d = cube
    # Print the shapes of the extracted images to verify
    print(img_3d.shape)  # Should be (64, 64, 64)
    ################################################################

    print()