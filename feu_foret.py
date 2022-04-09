import math

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

import random
import time

def prod_scalaire_2D(u, v):
    return u[0]*v[0] + u[1]*v[1]

"""
Cette version du feu de foret est optimisé pour tourner relativement rapidement même avec des gros feux.
La premiere approche moins optimisée et moins developée est présente dans le fichier "pasopti.py", 
et repose sur un parcourt intégral d'un numpy array representant chaque case de la foret. 
Ici cette approche repose sur le stockage des cases intéressantes dans des sets et des dictionnaires, par exemple
seules les cases en feu sont parcourut lors de "firestep", pour répendre le feu aux cases voisines.
La complexité dépend donc du nombre de case en feu et non du nombre de cases totales
"""

"""
Cases sans vegetation de base : void_tiles (set(cordonnées))
Cases en feu : fire_tiles. (dict {(cordonnées):nombre_de_tour_en_feu})
Cases brulés : burn_tiles (set(cordonnées))
Cases arbres : tree_tiles (dict {(cordonnées):coeff_brule}) avec coeff_brule entre 0 et 1.
"""

"""
Pompiers : Se déplacent suivant dfférentes stratégies et jetent de l'eau 
sur les arbres proches d'eux ce qui diminue de x% le coeff_brule des tree_tiles sur lequels ils sont.
ordre est l'endroit ou le pompier doit se diriger
Cases avec des pompiers : pomp_tiles (list [(cordonnées), ordre]) avec odre representant la case ou le pompier doit aller
dans ce programme les pompiers foncent sur le feu le plus proche d'eux
"""

"""
Pour lancer la simulation graphique :
start_graphic()
Pour lancer l'analyse :
start_analyse()

"""


taille = 500
Size = taille, taille
Mid = Size[0] // 2

tree_tiles = {}
void_tiles = set([])
def create_forest(p):
    """ Créer une foret avec une densité p d'arbre"""
    global void_tiles, tree_tiles
    tree_tiles = {}
    void_tiles = set([])
    for y in range(taille):
        for x in range(taille):
            randy = np.random.uniform(0,1)
            if randy > p:
                void_tiles.add((y, x))
            else:
                tree_tiles[(y, x)] = 1

def create_screen():
    global void_tiles, tree_tiles
    """ Créer l'image couleur qui sera affiché et modifié à chaque iteration"""
    screen = np.zeros((taille, taille, 3))
    for y in range(taille):
        for x in range(taille):
            if (y, x) in void_tiles:
                screen[y, x] = (0, 0, 0)
            else:
                screen[y, x] = (0, 0.8*tree_tiles[(y,x)], 1-tree_tiles[(y,x)]) #couleur des arbres. Si mouillé, donc coeff brulé plus bas, arbre plus sombre et bleu
    return screen


fire_tiles = {}
burn_tiles = set([])
new_fire_tiles = {}
new_burn_tiles = set([])
new_modif_tree = set([])
wet_tiles = set([])
pomp_tiles = []
time_test = 0
new_modif_void = set([])
screen = None


def create_new_simulation(p, nb_feu=20, graphic=True):
    """Créer une nouvelle simulation"""
    global fire_tiles, burn_tiles, screen, new_fire_tiles, new_burn_tiles, time_test, new_modif_tree, pomp_tiles, wet_tiles
    create_forest(p)
    screen = create_screen()
    fire_tiles = {}
    burn_tiles = set([])
    if graphic: screen = create_screen()
    new_fire_tiles = {}
    new_burn_tiles = set([])
    new_modif_tree = set([])
    wet_tiles = set([])
    pomp_tiles = []
    time_test = 0

    for i in range(nb_feu):  # créer une lignée d'arbres en feu
        fire_tiles[(Mid + i, Mid)] = 0


def fire_step(fire_prob, time_burn, r, vent):
    """Fonction s'occupant de la répartition du feu"""

    global fire_tiles, burn_tiles, new_fire_tiles, new_burn_tiles, void_tiles, time_test, tree_tiles
    #probabilité de contamination du feu, nombre de tour qu'un arbre met à bruler, rayon d'action d'un arbre en feu



    new_fire_tiles = {}
    new_burn_tiles = set([])


    for tile in fire_tiles:
        fire_tiles[tile] += 1
        if fire_tiles[tile] > time_burn:
            new_burn_tiles.add(tile)



        yt, xt = tile
        # Boucle créant les cases potentiellement en feu au prochain tour
        cords = []
        for a in range(-r, r + 1):
            for b in range(-r, r + 1):
                c = (yt+a, xt+b)
                if a * a + b * b <= r * r:
                    if 0<=c[0]<taille and 0<=c[1]<taille and c[0] and not(c in fire_tiles) and not(c in burn_tiles) and not(c in void_tiles):
                        cords.append(c)


        # Boucle parcourant les cases potentiellement en feu au prochain tour

        at = time.time()
        for c in cords:
            if c in new_fire_tiles:
                continue
            randy = np.random.uniform(0, 1)

            vec_prop = (c[0]-yt, c[1]-xt) #Vecteur representant la potentielle propagation du feu
            dist_s = vec_prop[0]**2 + vec_prop[1]**2 #distance au carré de la case potentiellement en feu et celle en feu
            vec_prop_unit = (vec_prop[0]*dist_s**-.5, vec_prop[1]*dist_s**-.5)
            prob_vent = vent[0]*vec_prop_unit[0] + vent[1]*vec_prop_unit[1] #Probabilité causé par le vent, produit scalaire
            #Si vec_prop et vent dans le même sens, probabilité de feu augmenté, sinon diminué.

            p = fire_prob*tree_tiles[c]*(1+prob_vent)/dist_s #probabilité modifié avec tout les autres facteurs
            if randy <= p:
                new_fire_tiles[c] = 0
                del tree_tiles[c]
                if c in wet_tiles:
                    wet_tiles.remove(c)
        time_test += time.time() - at


    burn_tiles = burn_tiles.union(new_burn_tiles)

    diff_keys = fire_tiles.keys() - new_burn_tiles
    fire_tiles = {k: fire_tiles[k] for k in diff_keys} # Fait la difference entre fire_tiles et dead_fire_tiles pour avoir les cases encore en feu
    fire_tiles.update(new_fire_tiles) # Fait l'union des deux dictionnaires

def new_pomp(cord, ordre=(Mid, Mid)):
    """Creer un pompier sur la case de cordonnées cord"""
    global pomp_tiles
    pomp_tiles.append([cord, ordre])

def pomp_step(r, p, v):
    """Fonction s'occupant de l'action des pompiers"""
    global fire_tiles, tree_tiles, pomp_tiles, new_modif_tree, wet_tiles, new_burn_tiles, prob_fire, new_modif_void
    new_pomp_tiles = []

    for pomp in pomp_tiles:
        pomp_pos = pomp[0]
        if pomp_pos in burn_tiles:
            new_burn_tiles.add(pomp_pos)
        if pomp_pos in void_tiles:
            new_modif_void.add(pomp_pos)


        yt, xt = pomp_pos
        for a in range(-r, r + 1):
            for b in range(-r, r + 1):
                c = (yt + a, xt + b)
                if a * a + b * b <= r * r:
                    if 0 <= c[0] < taille and 0 <= c[1] < taille and c[0] and c in tree_tiles:
                        tree_tiles[c] *= (1-p)
                        if tree_tiles[c] < 0.1:
                            tree_tiles[c] = 0.1
                        wet_tiles.add(c)
                        new_modif_tree.add(c)

        ordre = new_ordre_min(pomp)
        new_pos = deplacement_ordre(pomp, v)
        new_pomp_tiles.append([new_pos, ordre])

    pomp_tiles = new_pomp_tiles


def deplacement_ordre(pomp, vitesse):
    """Le pompier se déplace vers les cordonnées qui lui on été ordonnées"""
    pomp_pos, ordre = pomp
    if pomp_pos == ordre:
        return pomp_pos

    vec_diff = (ordre[0]-pomp_pos[0], ordre[1]-pomp_pos[1])

    norme_s = vec_diff[0]**2 + vec_diff[1]**2

    if norme_s == 0:
        return pomp_pos

    vec_diff_unit = (vitesse*vec_diff[0]*norme_s**-.5, vitesse*vec_diff[1]*norme_s**-.5)

    vec_diff_unit_int = [round(vec_diff_unit[0]), round(vec_diff_unit[1])]
    deplac = pomp_pos[0]+vec_diff_unit_int[0], pomp_pos[1]+vec_diff_unit_int[1]

    if 0<=deplac[0]<taille and 0<=deplac[1]<taille:
        return deplac

    return pomp_pos


def new_ordre_min(pomp):
    """le pompier se dirige vers la case en feu la plus proche de lui"""
    global fire_tiles
    if fire_tiles == {}:
        return pomp[0]

    pomp_pos,_ = pomp
    dist = lambda tile : (pomp_pos[0]-tile[0])**2 + (pomp_pos[1]-tile[1])**2
    min_fire = min(fire_tiles, key=dist)
    return min_fire

def evaporation(coeff_evapo):
    """evaporation des cases mouillés par les pompiers"""
    global wet_tiles, tree_tiles, new_modif_tree
    new_dry = set([])
    for wet in wet_tiles:
        tree_tiles[wet] *= coeff_evapo
        new_modif_tree.add(wet)
        if tree_tiles[wet] >= 1:
            tree_tiles[wet] = 1
            new_dry.add(wet)
    wet_tiles = wet_tiles-new_dry

def change_screen(screen):
    """Mets à jour la matrice représentant l'écran"""
    global new_fire_tiles, new_burn_tiles, new_modif_tree, tree_tiles, pomp_tiles, new_modif_void
    for burn_tile in new_burn_tiles:
        screen[burn_tile] = [0.2, 0.2, 0.2]
    for fire_tile in new_fire_tiles:
        screen[fire_tile] = [0.8, 0.2, 0.2]
    for tree_tile in new_modif_tree:
        screen[tree_tile] = [0, 0.8*tree_tiles[tree_tile], (1-tree_tiles[tree_tile])*0.5]
    for pomp in pomp_tiles:
        screen[pomp[0]] = [0, 1, 1]
    for void in new_modif_void:
        screen[void] = [0, 0, 0]
    new_modif_tree = set([])
    new_modif_void = set([])


def step(fire_args, pomp_args, evapo=True, pomp=True, graphic=True, coeff_evapo=1.01):
    """Passe la simulation de l'état n à n+1"""
    global time_test
    fire_step(*fire_args)
    if evapo: evaporation(coeff_evapo)
    if pomp: pomp_step(*pomp_args)
    if graphic: change_screen(screen)

    return fire_tiles == {} #Fin de la simulation si plus de casse qui brule, dans le mode analyse

compt = 0
def updatefig(*args):
    """Mets à jour l'image"""
    global screen, im, compt, graphic, fire_args, pomp_args
    compt += 1
    step(fire_args, pomp_args)
    im.set_array(screen)

    compt += 1
    return im,

def run_graphic():
    """Affiche l'animation"""
    global im
    fig = plt.figure()
    im = plt.imshow(screen, animated=True)
    ani = animation.FuncAnimation(fig, updatefig, interval=0,  blit=True)
    plt.show()

def start_graphic():
    global taille, Size, Mid, fire_args, pomp_args
    """Commence la simulation graphique"""

    # Le lecteur est libre de changer les parametres pour tester les différents scénarios

    taille = 500
    Size = taille, taille
    Mid = Size[0] // 2
    fire_args = (0.2, 1, 2, (0.3, 0))  # respectivement proba de transmission de feu, temps pour bruler, rayon de transmission de feu, vecteur vent
    # un vecteur vent de (0.3, 0) est un vent de force moyenne vers le bas
    pomp_args = (5, 0.3, 3)  # respectivement rayon d'action des pompiers, pouvoir des pompiers (c'est le x% décrit ligne 30), vitesse de déplacement des pompiers
    create_new_simulation(0.9)
    # Création des pompiers
    new_pomp((0, 0))
    new_pomp((taille, 0))
    new_pomp((0, taille))
    new_pomp((taille, taille))

    run_graphic()

# Analyse

def start_analyse():
    global taille, Size, Mid
    """Commence l'analyse"""
    taille = 100 #Taille de la grille determine la vitesse d'analyse mais aussi la précision
    Size = taille, taille
    Mid = Size[0] // 2

    nb_point = 50
    X = np.linspace(0, 1, nb_point)
    Y = []
    compt = 0
    for p in X:
        compt += 1
        """Le lecteur est libre de changer les parametres pour tester différents scénarios
        en ajoutant différents pompiers, en ajoutant du vent ou autre"""

        create_new_simulation(p, 10, graphic=False) #respectivement densité d'arbre, nombre de feu de départ
        fire_args = (0.2, 1, 3, (0, 0)) #respectivement proba de transmission de feu, temps pour bruler, rayon de transmission de feu, vecteur vent
        pomp_args = (10, 0.2, 3) #respectivement rayon d'action des pompiers, pouvoir des pompiers (c'est le x% décrit ligne 30), vitesse de déplacement des pompiers

        # new_pomp((0, 0))  #Ajout d'un pompier ou non
        while not(step(fire_args, pomp_args, graphic=False)):
            None
        print(compt/nb_point)
        prop_burn = 100*len(burn_tiles)/taille**2
        Y.append(prop_burn)


    plt.plot(X, Y)
    plt.ylim([0, 100])
    plt.show()








