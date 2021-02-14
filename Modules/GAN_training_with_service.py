import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
from IPython.display import clear_output


def plot_2(a, a_fake):
    plt.figure(figsize=(8, 4))

    a = np.array(a)
    a_fake = np.array(a_fake)

    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Input")
    plt.imshow(a, vmin=0, vmax=1)

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Output")
    plt.imshow(a_fake, vmin=0, vmax=1)
    plt.show()


def plot_4(a, a_fake, b, b_fake):
    plt.figure(figsize=(12, 8))

    a = np.array(a)
    a_fake = np.array(a_fake)
    b = np.array(b)
    b_fake = np.array(b_fake)

    plt.subplot(2, 2, 1)
    plt.axis("off")
    plt.title("A")
    plt.imshow(a, vmin=0, vmax=1)

    plt.subplot(2, 2, 2)
    plt.axis("off")
    plt.title("G(A)")
    plt.imshow(b_fake, vmin=0, vmax=1)

    plt.subplot(2, 2, 3)
    plt.axis("off")
    plt.title("F(B)")
    plt.imshow(a_fake, vmin=0, vmax=1)

    plt.subplot(2, 2, 4)
    plt.axis("off")
    plt.title("B")
    plt.imshow(b, vmin=0, vmax=1)
    plt.show();


def plot_6(a, a_fake, a_cycle, b, b_fake, b_cycle):
    plt.figure(figsize=(12, 8))

    a = np.array(a)
    a_fake = np.array(a_fake)
    b = np.array(b)
    b_fake = np.array(b_fake)

    plt.subplot(2, 3, 1)
    plt.axis("off")
    plt.title("A")
    plt.imshow(a, vmin=0, vmax=1)

    plt.subplot(2, 3, 2)
    plt.axis("off")
    plt.title("G(A)")
    plt.imshow(b_fake, vmin=0, vmax=1)

    plt.subplot(2, 3, 3)
    plt.axis("off")
    plt.title("F(G(A))")
    plt.imshow(a_cycle, vmin=0, vmax=1)

    plt.subplot(2, 3, 4)
    plt.axis("off")
    plt.title("F(B)")
    plt.imshow(a_fake, vmin=0, vmax=1)

    plt.subplot(2, 3, 5)
    plt.axis("off")
    plt.title("B")
    plt.imshow(b, vmin=0, vmax=1)

    plt.subplot(2, 3, 6)
    plt.axis("off")
    plt.title("G(F(B))")
    plt.imshow(b_cycle, vmin=0, vmax=1)
    plt.show();


def train(
        epochs,
        gan,
        gan_opt,
        gan_loader,
        device,
        criterion_1,
        criterion_2,
        model_name,
        path,
        info,
        D_mult=0.5,
        k_L1=10,
        k_L2=1,
        small_period=15,
        medium_period=40,
        silence=False,
        save=False,
        save_period=4000
):
    for epoch in range(1, epochs + 1):
        for i, data in enumerate(gan_loader.trainA):
            gan.level_up(data.shape[0])

            ## Data
            A_samples = data.to(device)
            B_samples = next(iter(gan_loader.trainB)).to(device)

            if (A_samples.shape[3] > 3) or (B_samples.shape[3] > 3):
                continue

            ### D_A real-samples
            gan.D_A.zero_grad()
            output = gan.D_A(A_samples.permute(0, 3, 1, 2)).view(-1)
            lossA_real = criterion_1(output, torch.ones(output.size()).to(device)) * D_mult
            lossA_real.backward()
            D_A_mean_for_real = output.mean().item()

            ### D_A fake-samples
            # print(B_samples.shape)
            A_fakes = gan.F(B_samples.permute(0, 3, 1, 2))
            output = gan.D_A(A_fakes.detach()).view(-1)
            lossA_fake = criterion_1(output, torch.zeros(output.size()).to(device)) * D_mult
            lossA_fake.backward()
            D_A_mean_for_fake = output.mean().item()
            lossD_A = lossA_real + lossA_fake
            gan_opt.D_A.step()

            ### D_B real-samples
            gan.D_B.zero_grad()
            output = gan.D_B(B_samples.permute(0, 3, 1, 2)).view(-1)
            lossB_real = criterion_1(output, torch.ones(output.size()).to(device)) * D_mult
            lossB_real.backward()
            D_B_mean_for_real = output.mean().item()

            ### D_B fake-samples
            B_fakes = gan.G(A_samples.permute(0, 3, 1, 2))
            output = gan.D_B(B_fakes.detach()).view(-1)
            lossB_fake = criterion_1(output, torch.zeros(output.size()).to(device)) * D_mult
            lossB_fake.backward()
            D_B_mean_for_fake = output.mean().item()
            lossD_B = lossB_real + lossB_fake
            gan_opt.D_B.step()

            ### A->G(A)->F(G(A)) and B->F(B)->G(F(B)) Cycle Consistency
            outputB = gan.D_B(B_fakes).view(-1)
            outputA = gan.D_A(A_fakes).view(-1)

            gan.G.zero_grad()
            gan.F.zero_grad()

            A_cycle = gan.F(B_fakes)
            B_cycle = gan.G(A_fakes)

            loss_G = criterion_1(outputA, torch.ones(outputA.size()).to(device)) + k_L1 * criterion_2(A_cycle,
                                                                                                      A_samples.permute(
                                                                                                          0, 3, 1, 2))
            loss_G += k_L2 ** criterion_2(B_fakes, A_samples.permute(0, 3, 1, 2))
            loss_F = criterion_1(outputB, torch.ones(outputB.size()).to(device)) + k_L1 * criterion_2(B_cycle,
                                                                                                      B_samples.permute(
                                                                                                          0, 3, 1, 2))
            loss_F += k_L2 ** criterion_2(B_fakes, B_samples.permute(0, 3, 1, 2))
            loss_GF = loss_G + loss_F

            loss_GF.backward()

            gan_opt.G.step()
            gan_opt.F.step()

            D_B_mean_for_fake_2 = output.mean().item()
            D_A_mean_for_fake_2 = output.mean().item()

            ### Save Losses
            info.loss_G.append(loss_G.item())
            info.loss_F.append(loss_F.item())
            info.loss_D_A.append(lossD_A.item())
            info.loss_D_B.append(lossD_B.item())
            info.acc_D_A_real.append(D_A_mean_for_real)
            info.acc_D_A_fake.append(D_A_mean_for_fake)
            info.acc_D_B_real.append(D_B_mean_for_real)
            info.acc_D_B_fake.append(D_B_mean_for_fake)

            if silence == False:
                ### Print Info
                if gan.level % small_period == 0:
                    print(f'It_ {gan.level}', end=', ')
                    print(f'Ep_ {epoch}/{epochs}', end='\n')

                    # print(f'Loss_G: {info.loss_G[-1]}', end=', ')
                    # print(f'Loss_D_B: {info.loss_D_B[-1]}', end=', ')
                    # print(f'D_B_mean_r: {D_B_mean_for_real}', end=', ')
                    # print(f'D_B_mean_f: {D_B_mean_for_fake}', end='\n')

                    # print(f'Loss_F: {info.loss_F[-1]}', end=', ')
                    # print(f'Loss_D_A: {info.loss_D_A[-1]}', end=', ')
                    # print(f'D_A_mean_r: {D_A_mean_for_real}', end=', ')
                    # print(f'D_A_mean_f: {D_A_mean_for_fake}', end='\n')

                ### Print figures
                if gan.level % medium_period == 0:
                    plt.plot(info.acc_D_A_real[-100:])
                    plt.plot(info.acc_D_A_fake[-100:])
                    plt.title('accuracy_D_A')
                    plt.show()

                    plt.plot(info.acc_D_B_real[-100:])
                    plt.plot(info.acc_D_B_fake[-100:])
                    plt.title('accuracy_D_B')
                    plt.show()

                ### Print images
                if gan.level % small_period == 0:
                    # print(torch.min(A_samples[0, :, :, :].detach().cpu()))
                    plot_6(A_samples[0, :, :, :].detach().cpu(),
                           A_fakes.permute(0, 2, 3, 1)[0, :, :, :].detach().cpu(),
                           A_cycle.permute(0, 2, 3, 1)[0, :, :, :].detach().cpu(),
                           B_samples[0, :, :, :].detach().cpu(),
                           B_fakes.permute(0, 2, 3, 1)[0, :, :, :].detach().cpu(),
                           B_cycle.permute(0, 2, 3, 1)[0, :, :, :].detach().cpu()
                           )
                if gan.level % (medium_period * 5) == 0:
                    clear_output(wait=True)

            if save == True:
                if gan.level % save_period == 0:
                    gan.save_weights(path, model_name, gan.level)