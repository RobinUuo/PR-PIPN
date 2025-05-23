import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

predicted_data = np.loadtxt('../predicted.txt')

xyz = predicted_data[:, :3]
pred_data = predicted_data[:, 3:]

with open('20.txt', 'r') as f:
    lines = [line.rstrip() for line in f if not line.startswith('%')]
data = np.array([line.split() for line in lines], dtype=float)

true_data = data[:, 3:]

pred_speed = np.linalg.norm(pred_data[:, :3], axis=1)
true_speed = np.linalg.norm(true_data[:, :3], axis=1)

pred_pressure = pred_data[:, 3]
true_pressure = true_data[:, 3]

pressure_error = np.abs(pred_pressure - true_pressure)
speed_error = np.abs(pred_speed - true_speed)



plotter = pv.Plotter(shape=(2, 3))


def add_scalar_plot(plotter, row, col, xyz, values, scalar_name, cmap):
    pdata = pv.PolyData(xyz)
    pdata[scalar_name] = values
    plotter.subplot(row, col)
    plotter.add_mesh(pdata, scalars=scalar_name, cmap=cmap, point_size=4,
                     show_scalar_bar=False, render_points_as_spheres=True)
    plotter.add_scalar_bar(f'{scalar_name}\n', vertical=False,
                           title_font_size=14, label_font_size=14,
                           fmt='%10.4f', n_labels=4, position_x=0.2)
    plotter.camera.zoom(0.75)


# 第一行：速度
add_scalar_plot(plotter, 0, 0, xyz, true_speed, 'True Speed', 'jet')
add_scalar_plot(plotter, 0, 1, xyz, pred_speed, 'Predicted Speed', 'jet')
add_scalar_plot(plotter, 0, 2, xyz, speed_error, 'Speed Error', 'cool')

# 第二行：压力
add_scalar_plot(plotter, 1, 0, xyz, true_pressure, 'True pressure', 'viridis')
add_scalar_plot(plotter, 1, 1, xyz, pred_pressure, 'Predicted Pressure', 'viridis')
add_scalar_plot(plotter, 1, 2, xyz, pressure_error, 'Pressure Error', 'cool')







plotter.show()
