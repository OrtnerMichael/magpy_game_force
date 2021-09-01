import numpy as np
import matplotlib.pyplot as plt

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
        sq = np.array([(odx,ody,0),(odx,-ody,0),(-odx,-ody,0),(-odx,ody,0),(odx,ody,0)])/2
        sq = orot.apply(sq) + ob.position
        plt.plot(sq[:,0],sq[:,1], color='k', lw=20)
    # figure style
    ax.axis('off')
    ax.set(xlim=(0, RESX), ylim=(0,RESY))
    plt.tight_layout()
    plt.savefig('bg', dpi=10)