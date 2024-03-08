import numpy as np 

# --------this code is copied from Alexander Fyrdahl:------
def biot_savart_simulation(segments, locations):
  """Reference : Esin Y, Alpaslan F, MRI image enhancement using Biot-Savart law at 3 tesla. Turk J Elec Eng & Comp Sci
  """

  num_coil_segments = segments.shape[0] - 1
  if num_coil_segments < 1:
      raise ValueError('Insufficient coil segments specified')

  if segments.shape[1] == 2:
      segments = np.hstack((segments, np.zeros((num_coil_segments, 1))))

  sensitivity_contribution = np.zeros((locations.shape[0], 3))

  segment_start = segments[0, :]
  for segment_index in range(num_coil_segments):
      segment_end = segment_start
      segment_start = segments[segment_index + 1, :]
      unit_segment_vector = (segment_end - segment_start) / np.linalg.norm(segment_end - segment_start)

      vector_u = -locations + segment_end
      vector_v = locations - segment_start

      cos_alpha = np.dot(vector_u, unit_segment_vector) / np.linalg.norm(vector_u, axis=1)
      cos_beta = np.dot(vector_v, unit_segment_vector) / np.linalg.norm(vector_v, axis=1)
      sin_beta = np.sin(np.arccos(cos_beta))

      sensitivity_magnitudes = (cos_alpha + cos_beta) / (np.linalg.norm(vector_v, axis=1) / sin_beta)

      cross_product_matrix = np.cross(np.identity(3), unit_segment_vector)
      normalized_sensitivity_directions = np.dot(cross_product_matrix, vector_v.T).T / np.linalg.norm(np.dot(cross_product_matrix, vector_v.T).T, axis=1)[:, np.newaxis]

      sensitivity_contribution += normalized_sensitivity_directions * sensitivity_magnitudes[:, np.newaxis]

  return np.linalg.norm(sensitivity_contribution, axis=1)

def define_coils(radius, center, pos, axis, segments=21):
  theta = np.linspace(0, 2 * np.pi, segments)
  if axis == 'x':
      x = np.full_like(theta, center[0] + pos)
      y = center[1] + radius * np.cos(theta)
      z = center[2] + radius * np.sin(theta)
  elif axis == 'y':
      x = center[0] + radius * np.cos(theta)
      y = np.full_like(theta, center[1] + pos)
      z = center[2] + radius * np.sin(theta)
  else:
      x = center[0] + radius * np.cos(theta)
      y = center[1] + radius * np.sin(theta)
      z = np.full_like(theta, center[2] + pos)
  return np.column_stack((x, y, z))

def compute_mri_coil_sensitivity(segments, locations, volume_shape, sphere):
  sensitivities = biot_savart_simulation(segments, locations)
  coil_image = np.zeros(volume_shape)
  coil_image[locations[:, 0], locations[:, 1], locations[:, 2]] = sensitivities
  return coil_image

def create_sphere_phantom(volume_shape=(192, 192, 192), radius=48):
  z, y, x = np.indices(volume_shape)
  center = np.array(volume_shape) // 2
  distance_from_center = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)
  return distance_from_center <= radius


# --------this code is copied from Alexander Fyrdahl:------