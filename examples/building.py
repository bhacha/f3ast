import f3ast
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


file_path = "SimpleCoil.stl"
stream_filename = "SimpleCoil"
scale_factor = 1
GR0 = .200 # in um/s, base growth rate
k = 1. #in 1/nm?, thermal conductivity 
sigma = 4 # in nm, dwell size




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



struct = f3ast.Structure.from_file(file_path, **settings["structure"])
rotation_axis, rotation_angle = (1, 0, 0), 90
tilt_string = "90t100"
struct.rotate(rotation_axis, rotation_angle)


struct.mirror(normal=(1, 0, 0))

struct.centre()  # centers xy to zero and sets minimum z value to zero
struct.rescale(scale_factor)  # scale the structure 3x


# with correction due to thermal conductivity
model = f3ast.DDModel(struct, GR0, k, sigma, **settings['dd_model'])

stream_builder, dwell_solver = f3ast.StreamBuilder.from_model(model, **settings['stream_builder'])
dwell_solver.print_total_time()


strm = stream_builder.get_stream()

# export with simple name

out_filename = f"{stream_filename}_{GR0*1000:.0f}gr_{k*100:.0f}k_{sigma:.0f}sig_{tilt_string}_{scale_factor}x"
strm.write(f"{out_filename}.str")