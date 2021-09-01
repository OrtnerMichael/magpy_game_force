import pygame
import numpy as np
import magpylib as magpy
import random
from physics import getFT, collision_xy_cuboids
from utility import create_bg_image

pygame.init()
pygame.display.set_caption("Magnetic Engine")
clock = pygame.time.Clock()

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0,0,255)
BLACK = (0,0,0)
ORANGE = (255, 136, 0)
GRAY = (200,200,200)

#############################################################
#############################################################
# SETTINGS
RESX, RESY = 960, 720
display_force = True
display_field_amp = True
display_field_lines = True
level = 3
massfactorF = 1000        # translation inertia = 1/mass
frictionF = 0.985         # translation friction
massfactorT = 1000        # rotation inertia = 1/moment of inertia
key_rot_vel = 5.5         # player key-rotation velocity 
maxboing = 5              # allowed wall crashes
player_size_x = 20
player_size_y = 20
player_start_vel = (1,-1,0)
nx, ny = 6, 6             # force computation discretization
M0 = 50                   # overall magnetization amplitude
#############################################################
#############################################################

# init target
target_radius = int(RESX*0.05)
target_pos = np.array((RESX-target_radius,target_radius,0))

# init player
start_pos_x = player_size_x + int(0.1*RESX)
start_pos_y = int(0.9*RESY) - player_size_y
player_pos = np.array((start_pos_x, start_pos_y, 0))
player_vel = np.array(player_start_vel, dtype=float)
player_vel_rot = 0
player_mag = np.array((M0,0,0), dtype=float)
player = magpy.magnet.Box(
    magnetization = player_mag,
    dimension = (2*player_size_x, 2*player_size_y, 20),
    position = player_pos)

pp = np.array([
    ( player_size_x, player_size_y,0), # Corner 1
    ( player_size_x,-player_size_y,0), # Corner 2
    (-player_size_x,-player_size_y,0), # Corner 3
    (-player_size_x, player_size_y,0), # Corner 4
    (-player_size_x/2,0,0),    # Arrow 1
    ( player_size_x/2,0,0),    # Arrow 2
    ( 0, player_size_y/4,0),   # Arrow 3
    ( 0,-player_size_y/4,0)])   # Arrow 4

# init obstactles
#    obstacle setup
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
    obst_rot = [0,90,180,270]*4
    obst_rot = random.sample(obst_rot, k=len(obst_possis))
elif level ==3:
    obst_dims = [[50,50,20]]*8
    obst_possis = np.array([
        (350,225,0), (350,475,0),
        (450,200,0), (450,500,0),
        (550,175,0), (550,525,0),
        (650,150,0), (650,550,0)])
    obst_mag = [(M0,0,0)]*8
    obst_rot = [-45,45]*4

#   create obst Magpylib Collection
obst = magpy.Collection()
for odim, opos, omag, orot in zip(obst_dims, obst_possis, obst_mag, obst_rot):
    obst + magpy.magnet.Box(
        magnetization = omag,
        dimension = odim,
        position = opos)
    obst[-1].rotate_from_angax(orot, 'z')

# init background image
if True:
    create_bg_image(RESX, RESY, display_field_amp, display_field_lines, obst)
bg = pygame.image.load("bg.png")
bg = pygame.transform.scale(bg, (RESX, RESY))
bg = pygame.transform.flip(bg, False, True)

# misc inits
screen = pygame.display.set_mode((RESX, RESY), 0, 0)
massfactorF /= player_size_x*player_size_y # scales with player size
massfactorT /= player_size_x*player_size_y # scales with player size
boings = 0            # boing counter
game_active = True    # game active
k_right = False       # right key
k_left = False        # left key
font = pygame.font.Font('freesansbold.ttf', 16)

############################################################
############################################################
while game_active:

    # key events
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


    # GAME PHYSICS ENGINE ####################################
    play_pos = player.position
    play_rot = player.orientation
    play_mag = play_rot.apply(player_mag)

    # compute 3D force and torque
    F, T = getFT([obst], [player], player.position)
    
    # avoid numerical cumulative problems
    F[2] = 0
    
    # apply force
    player_vel += F*massfactorF    # magneric acceleration
    player_vel *= frictionF        # air friction
    play_pos += player_vel
    player.position = play_pos
    
    # apply torque
    if not (k_right or k_left):
        player.rotate_from_angax(-T[2]*massfactorT*2000, 'z')

    # GRAPHIC DISPLAY ##################################
    screen.blit(bg, (0, 0))

    # draw player
    ppx = (play_rot.apply(pp) + play_pos)[:,:2]
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

    # draw target location
    pygame.draw.circle(screen, RED, target_pos[:2], target_radius, 6)

    # boing display
    text = font.render(f'Boings = {boings}', True, BLACK, WHITE)
    textRect = text.get_rect()
    textRect.center = ( 50, 20)
    screen.blit(text, textRect)

    # magnet collision
    collision = collision_xy_cuboids([player]*len(obst.sources), obst.sources)
    if isinstance(collision, int):
        print('CRASH !!!')
        print(collision)
        game_active = False

    # crash into walls    
    if (play_pos[0]<0) | (play_pos[0]>RESX):
        player_vel[0] = - player_vel[0]
        boings += 1
        print('BOING !!!')
    elif (play_pos[1]<0) | (play_pos[1]>RESY):
        player_vel[1] = - player_vel[1]
        boings += 1
        print('BOING !!!')
    if boings>maxboing:
        print('BOING + BOING + ... = CRASH !!!')
        game_active = False

    # victory
    if np.linalg.norm(play_pos-target_pos)<target_radius/2:
        print('YOU WIN !!!')
        game_active = False

    # UPDATE
    pygame.display.flip()

    # REFRESH
    clock.tick(120)

pygame.quit()