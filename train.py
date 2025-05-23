import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import matplotlib
import pandas as pd
import time
from model import Network
import matplotlib.pyplot as plt


matplotlib.use('TkAgg')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = '12'


class PhysicsLoss(nn.Module):
    def __init__(self, x_std, y_std):
        super(PhysicsLoss, self).__init__()
        self.x_std = x_std
        self.y_std = y_std

    def gradient(self, u_norm, x_norm, order=1, x_index=None, y_index=None):
        grad = u_norm
        for _ in range(order):
            grads = torch.autograd.grad(grad, x_norm, grad_outputs=torch.ones_like(grad),
                                        create_graph=True, retain_graph=True)[0]
            grad = grads[..., x_index]

        return grad * self.y_std[y_index] / self.x_std[x_index] ** order

    def forward(self, model, x_norm, nu, reduction='none'):
        out = model(x_norm)
        u_n, v_n, w_n, p_n = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]
        du_dx = self.gradient(u_n, x_norm, 1, 0, 0)
        dv_dy = self.gradient(v_n, x_norm, 1, 1, 1)
        dw_dz = self.gradient(w_n, x_norm, 1, 2, 2)
        continuity = du_dx + dv_dy + dw_dz

        dp_dx = self.gradient(p_n, x_norm, 1, 0, 3)
        dp_dy = self.gradient(p_n, x_norm, 1, 1, 3)
        dp_dz = self.gradient(p_n, x_norm, 1, 2, 3)

        d2u_dx2 = self.gradient(u_n, x_norm, 2, 0, 0)
        d2u_dy2 = self.gradient(u_n, x_norm, 2, 1, 0)
        d2u_dz2 = self.gradient(u_n, x_norm, 2, 2, 0)

        d2v_dx2 = self.gradient(v_n, x_norm, 2, 0, 1)
        d2v_dy2 = self.gradient(v_n, x_norm, 2, 1, 1)
        d2v_dz2 = self.gradient(v_n, x_norm, 2, 2, 1)

        d2w_dx2 = self.gradient(w_n, x_norm, 2, 0, 2)
        d2w_dy2 = self.gradient(w_n, x_norm, 2, 1, 2)
        d2w_dz2 = self.gradient(w_n, x_norm, 2, 2, 2)

        lap_u = d2u_dx2 + d2u_dy2 + d2u_dz2
        lap_v = d2v_dx2 + d2v_dy2 + d2v_dz2
        lap_w = d2w_dx2 + d2w_dy2 + d2w_dz2
        momentum_x = dp_dx - nu * lap_u
        momentum_y = dp_dy - nu * lap_v
        momentum_z = dp_dz - nu * lap_w

        loss_fn = nn.MSELoss(reduction='none')
        loss_continuity = loss_fn(continuity, torch.zeros_like(continuity))
        loss_momentum_x = loss_fn(momentum_x, torch.zeros_like(momentum_x))
        loss_momentum_y = loss_fn(momentum_y, torch.zeros_like(momentum_y))
        loss_momentum_z = loss_fn(momentum_z, torch.zeros_like(momentum_z))

        phy_loss_per_sample = loss_continuity + loss_momentum_x + loss_momentum_y + loss_momentum_z

        if reduction == 'mean':
            return phy_loss_per_sample.mean()
        elif reduction == 'none':
            return phy_loss_per_sample.squeeze()  # 从 [B,1] 转换为 [B]
        else:
            raise ValueError(f"Invalid reduction mode: {reduction}")


def compute_losses(model, phy_loss_fn,
                   xob_batch, xbc_batch, xdomain_batch, yob_batch,
                   nu, device, w_data=100, w_bc=1, w_pde=1, reduction='mean'):
    xob_batch = xob_batch.to(device)
    xbc_batch = xbc_batch.to(device)
    xdomain_batch = xdomain_batch.to(device)
    xdomain_batch.requires_grad_(True)
    yob_batch = yob_batch.to(device)
    # 预测和损失计算
    pred_ob = model(xob_batch)
    pred_bc = model(xbc_batch)

    ybc_n = -Y_mean[:3] / Y_std[:3]
    ybc_n = ybc_n.to(device)

    loss_data = torch.mean(torch.sum((pred_ob - yob_batch) ** 2, dim=1))
    loss_bc = torch.mean(torch.sum((pred_bc[:, :3] - ybc_n) ** 2, dim=1))
    loss_pde = phy_loss_fn(model, xdomain_batch, nu, reduction)

    total_loss = w_data * loss_data + w_bc * loss_bc + w_pde * loss_pde
    return total_loss, loss_data, loss_bc, loss_pde


def calArray2dDiff(array_0, array_1):
    array_0_rows = array_0.view([('', array_0.dtype)] * array_0.shape[1])
    array_1_rows = array_1.view([('', array_1.dtype)] * array_1.shape[1])

    return np.setdiff1d(array_0_rows, array_1_rows).view(array_0.dtype).reshape(-1, array_0.shape[1])


def farthest_point_sampling(xyz, n_samples):
    xyz = xyz.detach().cpu().numpy()
    farthest_pts = np.zeros((n_samples,), dtype=int)
    farthest_pts[0] = np.random.randint(len(xyz))
    distances = np.linalg.norm(xyz - xyz[farthest_pts[0]], axis=1)

    for i in range(1, n_samples):
        farthest_pts[i] = np.argmax(distances)
        dist_to_new = np.linalg.norm(xyz - xyz[farthest_pts[i]], axis=1)
        distances = np.minimum(distances, dist_to_new)
    return torch.tensor(farthest_pts, dtype=torch.long)


def load_data(file_name):
    input_file = file_name
    with open(input_file, 'r') as f:
        lines = [line.rstrip() for line in f if not line.startswith('%')]
    data = np.array([line.split() for line in lines], dtype=float)
    coord = data[:, :3]
    value = data[:, 3:]

    boundary_mask = np.all(value[:, :3] == 0, axis=1)
    internal_mask = ~boundary_mask

    boundary_data = data[boundary_mask]
    internal_data = data[internal_mask]

    num_obs = max(1, int(len(internal_data) * 0.05))
    coords = torch.tensor(internal_data[:, :3], dtype=torch.float32)
    fps_idx = farthest_point_sampling(coords, num_obs)
    obs_data = internal_data[fps_idx.numpy()]
    domain_data = calArray2dDiff(internal_data, obs_data)
    # plotter = pv.Plotter()
    # if boundary_data.size > 0:
    #     plotter.add_points(boundary_data[:, :3], color='gray', render_points_as_spheres=True, point_size=4)
    #
    # if obs_data.size > 0:
    #     plotter.add_points(obs_data[:, :3], color='yellow', render_points_as_spheres=True, point_size=4)
    #
    # if domain_data.size > 0:
    #     plotter.add_points(domain_data[:,:3], color='gray', render_points_as_spheres=True, point_size=4)
    #
    # plotter.set_background('white')
    # plotter.window_size = [800, 600]
    #
    # plotter.show()
    print('loading finished')
    return obs_data, domain_data, boundary_data


# ============== Training Loop ==============
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    file = '20.txt'
    model = Network()

    obs_data, domain_data, boundary_data = load_data(file)
    X_ob = torch.tensor(obs_data[:, :3], dtype=torch.float32)
    Y_ob = torch.tensor(obs_data[:, 3:], dtype=torch.float32)
    Xdomain = torch.tensor(domain_data[:, :3], requires_grad=True, dtype=torch.float32)
    X_bc = torch.tensor(boundary_data[:, :3], dtype=torch.float32)

    X_mean = X_ob.mean(0)
    X_std = X_ob.std(0)
    Y_mean = Y_ob.mean(0)
    Y_std = Y_ob.std(0)
    Xob_n = (X_ob - X_mean) / X_std
    Xdomain_n = (Xdomain - X_mean) / X_std
    Xbc_n = (X_bc - X_mean) / X_std
    yob_n = (Y_ob - Y_mean) / Y_std

    batch_size = 512
    min_lr = 1e-6
    obs_dataset = TensorDataset(Xob_n, yob_n)
    obs_loader = DataLoader(obs_dataset, batch_size=batch_size, shuffle=True)

    domain_loader = DataLoader(TensorDataset(Xdomain_n), batch_size=batch_size * 3, shuffle=True)
    bc_loader = DataLoader(TensorDataset(Xbc_n), batch_size=batch_size, shuffle=True)

    model = model.to(device)
    phy_loss = PhysicsLoss(x_std=X_std.to(device), y_std=Y_std.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     'min', factor=0.5, patience=1000,verbose=True,
                                                       min_lr=min_lr)
    epochs = 80000
    target_loss = 1e-6
    best_mse = float('inf')
    loss_history = []
    epoch_losses = []
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = epoch_data_loss = epoch_bc_loss = epoch_pde_loss = 0
        domain_iter = iter(domain_loader)
        bc_iter = iter(bc_loader)

        for (xob_batch, yob_batch) in obs_loader:
            try:
                xdomain_batch = next(domain_iter)[0].to(device)
                xbc_batch = next(bc_iter)[0].to(device)
            except StopIteration:
                domain_iter = iter(domain_loader)
                bc_iter = iter(bc_loader)
                xdomain_batch = next(domain_iter)[0].to(device)
                xbc_batch = next(bc_iter)[0].to(device)

            optimizer.zero_grad()
            w_p = 1 + (epoch / 20000) * 99 if epoch < 20000 else 100
            loss, loss_data, loss_bc, loss_pde = compute_losses(
                model, phy_loss, xob_batch, xbc_batch, xdomain_batch, yob_batch,
                nu=1e-3, device=device, w_pde=w_p, reduction='mean')

            loss.backward(retain_graph=True)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_data_loss += loss_data.item()
            epoch_bc_loss += loss_bc.item()
            epoch_pde_loss += loss_pde.item()

        scheduler.step(epoch_loss)
        loss_history.append([epoch, epoch_loss, epoch_data_loss, epoch_bc_loss, epoch_pde_loss])
        # ---- 每1000轮进行 PDF-RBA 重采样 ----
        if epoch > 20000 and epoch % 1000 == 0:
            model.eval()

            all_residuals = []
            for xdomain_batch in domain_loader:
                xdomain_batch = xdomain_batch[0].to(device)
                loss_pde = phy_loss(model, xdomain_batch, nu=1e-3, reduction='none')
                all_residuals.append(loss_pde.detach().cpu().numpy())

            residuals = np.concatenate(all_residuals)

            residuals = residuals - residuals.min()
            residuals = residuals / (residuals.max() + 1e-8)

            mu = 0.4
            sigma = 0.2
            weights = np.exp(-0.5 * ((residuals - mu) / sigma) ** 2)
            weights /= weights.sum()
            weights = torch.tensor(weights, dtype=torch.float32)
            # plt.figure(figsize=(6, 4))
            # plt.hist(weights, bins=100, log=True, color='skyblue', edgecolor='black')
            # plt.xlabel('Weight Value')
            # plt.ylabel('Frequency (log scale)')
            # plt.title(f'Weight Distribution at Epoch {epoch}')
            # plt.tight_layout()
            # plt.show()

            sampler = WeightedRandomSampler(
                weights=weights.to(device),
                num_samples=len(Xdomain_n),
                replacement=True
            )

            # 更新domain_loader
            domain_loader = DataLoader(
                TensorDataset(Xdomain_n),
                batch_size=batch_size * 3,
                sampler=sampler,
                shuffle=False,
                pin_memory=True
            )

        if epoch % 100 == 0:
            print(f"[{epoch}/{epochs}]: Total={epoch_loss:.3e}, "
                  f"Data={epoch_data_loss:.3e}, "
                  f"BC={epoch_bc_loss:.3e},"
                  f"PDE={epoch_pde_loss:.3e}")

        if epoch_loss < best_mse:
            best_mse = epoch_loss
            torch.save(model.state_dict(), '../best_model.pth')

        if epoch_loss <= target_loss or optimizer.param_groups[0]['lr'] <= min_lr:
            print(f'Training stopped at epoch {epoch + 1}')
            break

    end_time = time.time()
    print(f'Training time: {int(end_time - start_time) / 60} minutes')

    with open(file, 'r') as f:
        lines = [line.rstrip() for line in f if not line.startswith('%')]
    full_data = np.array([line.split() for line in lines], dtype=float)
    coords = torch.tensor(full_data[:, :3], dtype=torch.float32)
    coords_n = (coords - X_mean) / X_std
    model.load_state_dict(torch.load("../best_model.pth"))
    model.eval()
    with torch.no_grad():
        pred_norm = model(coords_n.to(device)).cpu().numpy()
        pred_phys = pred_norm * Y_std.numpy() + Y_mean.numpy()
        predicted_domain = np.hstack((coords, pred_phys))
        np.savetxt("../predicted.txt", predicted_domain, fmt="%.6f")
    df = pd.DataFrame(loss_history, columns=['Epoch', 'TotalLoss', 'PhysLoss', 'DataLoss', 'BCLoss'])
    df.to_csv('loss_history.csv', index=False)

    plt.figure(figsize=(12, 4))
    plt.plot(df['Epoch'], df['TotalLoss'], label='Total Loss')
    plt.plot(df['Epoch'], df['PhysLoss'], label='Physics Loss')
    plt.plot(df['Epoch'], df['DataLoss'], label='Data Loss')
    plt.plot(df['Epoch'], df['BCLoss'], label='BC Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PINN Training Losses')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=300)
    plt.show()
