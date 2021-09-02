import numpy as np
import pygame
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_bg_image(
    RESX,
    RESY,
    display_field_amp,
    display_field_lines,
    obst):
    # create a grid in the xz-symmetry plane
    xs = np.linspace(0, RESX, 45)
    ys = np.linspace(0, RESY, 45)
    grid = np.array([[(x,y,0) for x in xs] for y in ys])
    # compute B field on grid using a source method
    # display field with Pyplot
    fig, ax = plt.subplots(1, 1, figsize=(RESX/10,RESY/10))
    if display_field_amp | display_field_lines:
        Bgrid = obst.getB(grid)
    if display_field_amp:
        ampBgrid = np.linalg.norm(Bgrid, axis=2)
        ax.contourf(grid[:,:,0], grid[:,:,1], np.log(ampBgrid), 100, cmap='rainbow')
    if display_field_lines:
        ax.streamplot(grid[:,:,0], grid[:,:,1], Bgrid[:,:,0], Bgrid[:,:,1],
            density=3, color='k', linewidth=10, arrowsize=15)
    # plot obstacle boundaries
    for ob in obst:
        orot = ob.orientation
        odx,ody,_ = ob.dimension
        sq = np.array([(odx,ody,0),(odx,-ody,0),(-odx,-ody,0),(-odx,ody,0)])/2
        sq = orot.apply(sq) + ob.position
        arr = np.array([(-odx/4,0,0), (odx/4,0,0), (0,ody/8,0), (odx/4,0,0), (0,-ody/8,0)])
        arr = orot.apply(arr) + ob.position
        #plt.plot(sq[:,0],sq[:,1], color='k', lw=20)
        # Create a Rectangle patch
        rect = patches.Polygon(sq[:,:2], linewidth=20, edgecolor='k', facecolor=(.3,.3,.3), zorder=9)
        plt.plot(arr[:2,0], arr[:2,1], color='w', lw=20, zorder=10)
        plt.plot(arr[2:,0], arr[2:,1], color='w', lw=20, zorder=10)
        ax.add_patch(rect)
    # figure style
    ax.axis('off')
    ax.set(xlim=(0, RESX), ylim=(0,RESY))
    plt.tight_layout()
    plt.savefig('bg', dpi=10)


def draw_player(player, screen, display_force, F):
    """
    draw player magnet on screen
    """
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLACK = (0,0,0)
    ORANGE = (255, 136, 0)
    GRAY = (200,200,200)

    # compute player points on screen
    dimx, dimy, _ = player.dimension
    pp = np.array([
        ( dimx/2, dimy/2,0), # Corner 1
        ( dimx/2,-dimy/2,0), # Corner 2
        (-dimx/2,-dimy/2,0), # Corner 3
        (-dimx/2, dimy/2,0), # Corner 4
        (-dimx/4,0,0),    # Arrow 1
        ( dimx/4,0,0),    # Arrow 2
        ( 0, dimy/8,0),   # Arrow 3
        ( 0,-dimy/8,0)])   # Arrow 4

    # draw player
    play_pos = player.position
    ppx = (player.orientation.apply(pp) + play_pos)[:,:2]
    pygame.draw.polygon(screen, GRAY, [ppx[0],ppx[1],ppx[2],ppx[3]])
    pygame.draw.line(screen, GREEN, ppx[0], ppx[1], 5)
    pygame.draw.line(screen, RED, ppx[2], ppx[3], 5)
    pygame.draw.line(screen, BLACK, ppx[1], ppx[2], 3)
    pygame.draw.line(screen, BLACK, ppx[3], ppx[0], 3)
    pygame.draw.line(screen, BLACK, ppx[4],ppx[5], 3)
    pygame.draw.line(screen, BLACK, ppx[5],ppx[6], 3)
    pygame.draw.line(screen, BLACK, ppx[5],ppx[7], 3)

    # display force vector
    if display_force:
        F_len = np.linalg.norm(F)
        F_norm = F/F_len
        pygame.draw.line(screen, ORANGE, play_pos[:2], play_pos[:2]+F_norm[:2]*(10+F_len*.1), 3)
