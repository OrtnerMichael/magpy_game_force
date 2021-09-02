import pygame
import numpy as np
import magpylib as magpy
import random
from physics import getFT, collision_xy_cuboids
from utility import create_bg_image, draw_player

pygame.init()
pygame.display.set_caption("Magnetic Engine")
clock = pygame.time.Clock()




#############################################################
#############################################################
# SETTINGS
RESX, RESY = 960, 720
display_force = True
display_field_amp = True
display_field_lines = True
level = 2
massfactorF = 800        # translation inertia = 1/density
frictionF = 0.985       # translation friction
massfactorT = 800        # rotation inertia = 1/moment of inertia (density)
key_rot_vel = 5.5         # player key-rotation velocity 
player_size = (40, 30, 20)
player_start_pos = (116, 628, 0)
player_start_vel = (1,-1,0)
nx, ny = 6, 6             # force computation discretization
M0 = 50                   # overall magnetization amplitude
#############################################################
#############################################################


# GAME INITIALIZATION #######################################

# init target
target_radius = int(RESX*0.05)
target_pos = np.array((RESX-target_radius,target_radius,0))

# init player
player_vel = np.array(player_start_vel, dtype=float)
player_vol = np.prod(player_size)/200
player = magpy.magnet.Box(
    magnetization = (M0,0,0),
    dimension = player_size,
    position = player_start_pos)

# obstacle setups
if level == 1:
    obst_dims = [[int(RESX/8)]*3]
    obst_possis = np.array([(RESX/2, RESY/2, 0)])
    obst_mag = [(M0,0,0)]
    obst_rot = [0]
elif level == 2:
    obst_dims = np.random.rand(6,3)*60+30
    obst_dims[:,2] = 20
    obst_possis = np.array([
        (150, 375, 0), (450, 225, 0), (750,75,0),
        (375, 630, 0), (630, 525, 0), (900,300,0)])
    obst_mag = [(M0,0,0)]*6
    obst_rot = np.random.rand(6)*360

# init obstacles
obst = magpy.Collection()
for odim, opos, omag, orot in zip(obst_dims, obst_possis, obst_mag, obst_rot):
    obst + magpy.magnet.Box(
        magnetization = omag,
        dimension = odim,
        position = opos)
    obst[-1].rotate_from_angax(orot, 'z')

# create/load background image
if True:
    create_bg_image(RESX, RESY, display_field_amp, display_field_lines, obst)
bg = pygame.image.load("bg.png")
bg = pygame.transform.scale(bg, (RESX, RESY))
bg = pygame.transform.flip(bg, False, True)

# misc inits
screen = pygame.display.set_mode((RESX, RESY), 0, 0)
massfactorF /= player_vol # scales with player size
massfactorT /= player_vol # scales with player size
boings = 0            # boing counter
game_active = True    # game active
k_right = False       # right key
k_left = False        # left key
font = pygame.font.Font('freesansbold.ttf', 16)

############################################################
############################################################
while game_active:

    # KEY EVENTS #######################################
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_active = False
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                k_right = True
            elif event.key == pygame.K_LEFT:
                k_left = True
        
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_RIGHT:
                k_right = False
            elif event.key == pygame.K_LEFT:
                k_left = False

    if k_right:
        player.rotate_from_angax(key_rot_vel, 'z')
    elif k_left:
        player.rotate_from_angax(-key_rot_vel, 'z')


    # GAME PHYSICS ENGINE ###############################
    # compute 3D force and torque
    F, T = getFT([obst], [player], player.position)
    # avoid numerical cumulative problems
    F[2] = 0
    
    # apply force (inertia + friction)
    player_vel += F*massfactorF    # magneric acceleration
    player_vel *= frictionF        # air friction
    player.position += player_vel
    
    # apply torque (no inertia = 100% friction)
    if not (k_right or k_left):
        player.rotate_from_angax(-T[2]*massfactorT*2000, 'z')


    # GRAPHIC DISPLAY ###################################
    screen.blit(bg, (0, 0))

    draw_player(player, screen, display_force, F)

    # draw target location
    pygame.draw.circle(screen, (255, 0, 0), target_pos[:2], target_radius, 6)


    # GAME CONDITIONS ###################################
    # magnet collision
    collision = collision_xy_cuboids([player]*len(obst.sources), obst.sources)
    if isinstance(collision, int):
        print('CRASH !!!')
        print(collision)
        game_active = False

    # wall boings
    play_pos = player.position
    if (play_pos[0]<0) | (play_pos[0]>RESX):
        player_vel[0] = - player_vel[0]
        boings += 1
        print('BOING !!!')
    elif (play_pos[1]<0) | (play_pos[1]>RESY):
        player_vel[1] = - player_vel[1]
        boings += 1
        print('BOING !!!')

    # victory
    if np.linalg.norm(play_pos-target_pos)<target_radius/2:
        print('YOU WIN !!!')
        game_active = False


    # UPDATE ##############################################
    pygame.display.flip()
    clock.tick(120)

pygame.quit()