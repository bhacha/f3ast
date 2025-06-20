import f3ast
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# settings can be defined manually in a dictionary:
settings = {}
settings["structure"] = {"pitch": 3, "fill": False}  # in nm
settings["stream_builder"] = {
    "addressable_pixels": [65536, 56576],
    "max_dwt": 5,  # in ms
    "cutoff_time": 0.01,  # in ms, for faster exporting: remove dwells below cutoff time
    "screen_width": 10.2e3,  # in nm, horizontal screen width / field of view
    # 'serpentine' or 'serial', scanning order between slices
    "scanning_order": "serpentine",
}
# pixel size for thermal resistance
settings["dd_model"] = {"single_pixel_width": 50}

file_path = "f3ast/examples/SimpleCoil.stl"
struct = f3ast.Structure.from_file(file_path, **settings["structure"])

# rotate: specify axis and angle in degrees
# this is for example useful if FEBID growth is done with a tilted SEM sample stage
rotation_axis, rotation_angle = (1, 0, 0), 90
struct.rotate(rotation_axis, rotation_angle)

# In some cases (e.g. Helios 600) , stream files appear mirrored on the SEM screen
# compared to the orientation of the initial stl structure.
# If precise orientation of the structure (e.g. with respect to the GIS) is important,
# a mirror operation can be applied across a plane with a given normal.
struct.mirror(normal=(1, 0, 0))

struct.centre()  # centers xy to zero and sets minimum z value to zero
struct.rescale(1)  # scale the structure 3x

# # interactive plot for inspection
# struct.show()


GR0 = 50e-3 # in um/s, base growth rate
k = 1. # in 1/nm?, thermal conductivity 
sigma = 4.4 # in nm, dwell size

# with correction due to thermal conductivity
model = f3ast.DDModel(struct, GR0, k, sigma, **settings['dd_model'])

ax, sc= struct.plot_slices(c=np.concatenate(model.resistance), cmap="hot")
plt.show(block=True)