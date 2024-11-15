import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('qtagg')
'''
Open a stream file and plot the data

Stream files are formatted: dwell, x (pixels), y (pixels)
'''

#!%matplotlib
file_name = 'streams/Cyl_Test_bigsingle_200gr_15k_4sig_p3'

streamlist = []


with open(file_name+'.str', "r") as f:
    for line in f:
        streamlist.append(line.rstrip("\n"))




def downsample(streamlist, factor):
    num_points = int(streamlist[2])
    downsampled_steps = round(num_points/(factor*num_points))
    new_points = range(3, num_points, downsampled_steps)
    xpos = []
    ypos = []
    dwell_time = []
    for k in new_points:
        streamline = streamlist[k]
        if len(streamline) > 8:
            # Ignore header lines
            streampoints = streamline.split()
            xpos.append(int(streampoints[1]))
            ypos.append(int(streampoints[2]))
            dwell_time.append(int(streampoints[0]))
    return xpos, ypos, dwell_time, new_points

xpos, ypos, dwell_time, new_points = downsample(streamlist, 1)


ordering = range(0,len(dwell_time))

print(len(new_points))


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(xpos, ypos, new_points)


def output_stream(filename, xpos, ypos, dwell_time):
    numpoints = str(len(xpos))
    with open(filename+'.str', 'w') as f:
        f.write('s16\n1\n')
        f.write(numpoints+'\n')
        for k in range(int(numpoints)):
            xstring = str(xpos[k])
            ystring = str(ypos[k])
            dwellstring = str(dwell_time[k])
            linestring = xstring+" "+ystring+" "+dwellstring
            f.write(linestring + '\n')


# output_stream('teststream', xpos, ypos, dwell_time)