# from tvtk.api import tvtk, write_data
import numpy as np
# from utils.evaluate_utils import get_boundaries, calculate_mean_speed
import matplotlib.pyplot as plt
import h5py

if __name__ == '__main__':
    in_vivo_path = '/home/pcallmer/Temporal4DFlowNet/data/CARDIAC/M1_2mm_step2_static_dynamic_noise.h5'
    data_original = {}
    vel_colnames = ['u', 'v','w']
    venc_colnames = [ 'u_max', 'v_max', 'w_max']
    mag_colnames = [ 'mag_u', 'mag_v', 'mag_w']
    vencs = {}

    a = np.array([0, 1, 0])
    print(np.equal(a, 1.0))
    print("test script run")
    if False:
        for m in ['1', '2', '3', '4']:
            in_vivo_path = f'/home/pcallmer/Temporal4DFlowNet/data/CARDIAC/M{m}_2mm_step2_static_dynamic.h5'
            with h5py.File(in_vivo_path, mode = 'r' ) as p1:
                        # print(p1.keys())
                        print('M', m)
                        print('shape', p1['u'].shape)
                        print(np.array(p1['dx']))
                        mask =  np.asarray(p1['mask'])
                        temporal_mask = mask.copy()

                        # data_original['mask'] = temporal_mask
                        for vel, venc, mag in zip(vel_colnames, venc_colnames, mag_colnames):
                            data_original[vel] = np.asarray(p1[vel])
                            data_original[f'{vel}_fluid'] = np.multiply(data_original[vel], temporal_mask)
                        #     data_original[mag] = np.asarray(p1[mag])

                        speed = np.sqrt(np.square(data_original['u']) + np.square(data_original['v']) + np.square(data_original['w']))
                        mean_speed = calculate_mean_speed(data_original['u'], data_original['v'], data_original['w'], temporal_mask)
                        plt.clf()
                        plt.figure(figsize=(10,3))
                        plt.plot(mean_speed, '.-', label = 'High resolution', color = 'black')
                        plt.title('Mean speed')
                        plt.xlabel('frame')
                        plt.ylabel(' mean speed (cm/s)')
                        plt.legend()
                        plt.savefig(f'/home/pcallmer/Temporal4DFlowNet/results/data/mean_speed_M{m}.svg')

                        # print('mean speed ', mean_speed)
                        # print('mean speed ', np.average(mean_speed))
                        # print('mean speed ', np.average(speed, weights=temporal_mask, axis = (1,2,3))*100)
                        # print('mean speed diff', np.average(speed, weights=temporal_mask, axis = (1,2,3))*100 - mean_speed)
                        print('mean speed min', np.min(mean_speed))
                        print('mean speed max', np.max(mean_speed))
                        print('mean speed mean', np.mean(mean_speed))
                        print('mean speed std', np.std(mean_speed))
                        print('mean speed median', np.median(mean_speed))

                        print('max speed ', np.max(speed))
                        print('min speed ', np.min(speed))
                        print('mean speed ', np.mean(speed))

   