import numpy as np, os, sys, tools21cm as t2c

from tqdm import tqdm
from datetime import datetime
from clump_functions import SameValuesInArray, SaveMatrix
from scipy import ndimage
from glob import glob

"""
args = np.array(sys.argv)
sim  = int(args[np.argwhere(args=='-boxsize')+1]) if '-boxsize' in args else 500
xfrac_file = str(args[np.argwhere(args=='-xfrac_file')+1]) if '-xfrac_file' in args else '/disk/dawn-1/garrelt/Reionization/C2Ray_WMAP7/500Mpc/500Mpc_z50_0_300/results/xfrac3d_9.308.bin'
dens_file = str(args[np.argwhere(args=='-dens_file')+1]) if '-dens_file' in args else '/disk/dawn-1/garrelt/Reionization/C2Ray_WMAP7/500Mpc/coarser_densities/nc300/9.308n_all.dat'
out_dir  = str(args[np.argwhere(args=='-out_dir')+1]) if '-out_dir' in args else './'
out_name = str(args[np.argwhere(args=='-out_name')+1]) if '-out_name' in args else 'image_%d'%time.time()
out_res  = int(args[np.argwhere(args=='-out_res')+1]) if '-out_res' in args else 64
"""

sim = 64
mesh = 256
smt = True
'''
# 500 Mpc
path_xi = '/research/prace3/reion/500Mpc_RT/500Mpc_f5_8.2pS_300_stochastic_Cmean/results/'
path_dens = '/research/prace/sph_smooth_cubepm_130627_12_6912_500Mpc_ext2/global/so/nc300/'

# 244 Mpc
path_xi = '/research/prace/244Mpc_f2_8.2pS_500/results/'
path_dens = '/research/prace/sph_smooth_cubepm_130329_10_4000_244Mpc_ext2_test/global/so/nc500/'
'''
# 64 Mpc
path_xi = '/research/prace/64Mpc_f2_8.2pS_256/results/'
path_dens = '/research/prace/sph_smooth_cubepm_130708_8_2048_64Mpc_ext2/nc256/'


out_res = 64
out_dir = './2DlogPk-%dx%d_%s_%dMpc' %(out_res, out_res, datetime.now().strftime('%d%m%y'), sim)
os.makedirs(out_dir)

# interpolate redshift list
redshift_xi = t2c.get_xfrac_redshifts(path_xi)
redshift_dens = t2c.get_dens_redshifts(path_dens)
redshift = SameValuesInArray(redshift_xi, redshift_dens)

print('tot redshift:', redshift.size)
#### Code below this is important
t2c.set_sim_constants(sim)

min_arr = []
max_arr = []

for i in tqdm(range(len(redshift))):
        z = redshift[i]
        xf = t2c.XfracFile('%sxfrac3d_%.3f.bin' %(path_xi, z)).xi
        dn = t2c.DensityFile('%s%.3fn_all.dat' %(path_dens, z)).cgs_density
        dt = t2c.calc_dt(xf, dn, z)
        ps_dt = t2c.power_spectrum_nd(dt)
	if(smt): ps_dt = ndimage.gaussian_filter(ps_dt, 5, mode='wrap')

	for ax in ['x', 'y', 'z']:
		if(ax=='x'):
			rs_dt = np.log10(ps_dt[:, int(mesh/2):int(mesh/2)+out_res, int(mesh/2):int(mesh/2)+out_res])
		elif(ax=='y'):
			rs_dt = np.log10(ps_dt[int(mesh/2):int(mesh/2)+out_res, :, int(mesh/2):int(mesh/2)+out_res])
		else:
			rs_dt = np.log10(ps_dt[int(mesh/2):int(mesh/2)+out_res, int(mesh/2):int(mesh/2)+out_res, :])
		
	        for j in range(len(rs_dt)):
			if(ax=='x'):
				pp = rs_dt[j, :, :]
			elif(ax=='y'):
				pp = rs_dt[:, j, :]
			else:
				pp = rs_dt[:, :, j]

	                SaveMatrix('%s/image_%si%dj%d_%dMpc.bin' %(out_dir, ax, i, j, sim), pp, (out_res, out_res))
	                min_arr.append(np.min(pp))
	                max_arr.append(np.max(pp))

print('tot max and min:\t', np.max(max_arr), np.min(min_arr))
print('tot number of data:\t', len(glob('%s/*.bin' %out_dir)))

