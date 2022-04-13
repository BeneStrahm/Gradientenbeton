# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Description:  Berechnung von Masse-, Querschnitts- und Steifigkeits-
#               reduktionsfaktoren von mit kugelförmigen Hohlkörpern gradierten
#               Betondecken anhand einer Einheitszelle
# Author:       benedikt.strahm@ilek.uni-stuttgart.de
# Created:      09.02.2022
# Execution:    Executing from command line (py filename.py)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Libraries
# ------------------------------------------------------------------------------
import numpy as np
from scipy.integrate import quad

# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------


def radiusAtSection(D_hk, x, e):
    """Radius am betrachteten Schnitt entlang x-Achse
    :param D_hk: Hohlkörperdurchmesser in cm
    :param x: Stelle des betrachteten Schnittes entlang x-Achse in cm
    :param e: Abstand Hohlkörper in Querrichtung in cm
    :rtype r_x: Radius am betrachteten Schnitt in cm
    """
    r_hk = D_hk / 2

    # Wenn innerhalb Hohlkörperbereich
    if x <= r_hk:
        r_x = np.sqrt(r_hk ** 2 - x ** 2)

    # Wenn innerhalb Massivbereich der Einheitszelle
    elif (x >= r_hk) and (x <= r_hk + e / 2):
        r_x = 0

    # Wenn außerhalb der Einheitszelle - Fehler
    else:
        print("Betrachteter Schnitt liegt außerhalb der Einheitszelle")

    return r_x


def crossSection(r_x, h, D_hk, e):
    """Querschnittflächen am betrachteten Schnitt entlang x-Achse
    :param r_x: Radius am betrachteten Schnitt in cm
    :param h: Deckenstärke in cm
    :param D_hk: Hohlkörperdurchmesser in cm
    :param e: Abstand Hohlkörper in Querrichtung in cm
    :rtype A_brt: Bruttoquerschnitt in cm^2
    :rtype A_hk: Hohlkörperquerschnitt in cm^2
    :rtype A_net: Nettoquerschnitt in cm^2
    """
    A_brt = h * (D_hk + e)
    A_hk = np.pi * r_x ** 2
    A_net = A_brt - A_hk

    return A_brt, A_hk, A_net


def centerOfGravity(h, D_hk, d_OK, A_brt, A_hk, A_net):
    """Schwerpunkte am betrachteten Schnitt entlang x-Achse
    :param h: Deckenstärke in cm
    :param D_hk: Hohlkörperdurchmesser in cm
    :param d_OK: obere Deckschicht in cm
    :param A_brt: Bruttoquerschnitt in cm^2
    :param A_hk: Hohlkörperquerschnitt in cm^2
    :param A_net: Nettoquerschnitt in cm^2
    :rtype zs_brt: Bruttoquerschnitt in cm
    :rtype zs_hk: Hohlkörperquerschnitt in cm
    :rtype zs_net: Nettoquerschnitt in cm
    """
    zs_brt = h / 2
    zs_hk = d_OK + D_hk / 2
    zs_net = (A_brt * zs_brt - A_hk * zs_hk) / A_net

    return zs_brt, zs_hk, zs_net


def momentOfInertia(r_x, h, D_hk, e, A_brt, A_hk, zs_brt, zs_hk, zs_net):
    """Flächenträgheitsmomente am betrachteten Schnitt entlang x-Achse
    :param r_x: Radius am betrachteten Schnitt in cm
    :param h: Deckenstärke in cm
    :param D_hk: Hohlkörperdurchmesser in cm
    :param e: Abstand Hohlkörper in Querrichtung in cm
    :param A_brt: Bruttoquerschnitt in cm^2
    :param A_hk: Hohlkörperquerschnitt in cm^2
    :param zs_brt: Bruttoquerschnitt in cm
    :param zs_hk: Hohlkörperquerschnitt in cm
    :param zs_net: Nettoquerschnitt in cm
    :rtype I_brt: Bruttoquerschnitt in cm^4
    :rtype I_hk: Hohlkörperquerschnitt in cm^4
    :rtype I_net: Nettoquerschnitt in cm^4
    """
    # Flächenträgheitsmoment des Bruttoquerschnittes
    I_brt = (D_hk + e) * h ** 3 / 12

    # .. bezogen auf den Schwerpunkt des Nettoquerschnittes
    I_s_brt = I_brt + A_brt * (zs_net - zs_brt) ** 2

    # Flächenträgheitsmoment des Hohlkörpers
    I_hk = np.pi * r_x ** 4 / 4

    # .. bezogen auf den Schwerpunkt des Nettoquerschnittes
    I_s_hk = I_hk + A_hk * (zs_net - zs_hk) ** 2

    # Flächenträgheitsmoment des Nettoquerschnittes
    I_s_net = I_s_brt - I_s_hk

    return I_brt, I_s_net


def integrandCrossSection(x, D_hk, e, h, d_OK):
    # Radius
    r_x = radiusAtSection(D_hk, x, e)

    # Querschnittsflächen
    A_brt, A_hk, A_net = crossSection(r_x, h, D_hk, e)

    return A_net  # I_brt, I_s_net


def integrandMomentOfInertia(x, D_hk, e, h, d_OK):
    # Radius
    r_x = radiusAtSection(D_hk, x, e)

    # Querschnittsflächen
    A_brt, A_hk, A_net = crossSection(r_x, h, D_hk, e)

    # Schwerpunkte
    zs_brt, zs_hk, zs_net = centerOfGravity(
        h, D_hk, d_OK, A_brt, A_hk, A_net)

    # Trägheitsmoment
    I_brt, I_s_net = momentOfInertia(
        r_x, h, D_hk, e, A_brt, A_hk, zs_brt, zs_hk, zs_net)

    return I_s_net  # I_brt, I_s_net


def getValidInput(prompt, a=float('-inf'), b=float('inf')):
    validnumber = False
    while not validnumber:
        try:
            inp = float(input(prompt))
            if b >= inp >= a:
                validnumber = True
            else:
                print('Ungültige Angabe, bitte versuchen Sie es noch einmal')
        except ValueError:
            print('Das ist keine Zahl, bitte versuchen Sie es noch einmal')
    return inp


def main():
    # Eingabeparameter
    # ----------------------
    print("Eingabeparameter\n----------------------")
    h = getValidInput("Deckenstärke in cm: ", 0)
    D_hk = getValidInput("Hohlkörperdurchmesser in cm: ", 0, h)
    d_UK = getValidInput("Stärke der unteren Deckschicht in cm: ", 0, h-D_hk)
    e = getValidInput("Hohlkörperabstand in Querrichtung in cm: ", 0)
    rho = getValidInput("Rohdichte Beton in kg/m³: ", 0)

    # Berechnung Geometrie
    # ----------------------
    d_OK = h - d_UK - D_hk  # obere Deckschicht in cm
    e2 = e + D_hk           # Achsabstand Hohlkörper in Querrichtung in cm
    r_hk = D_hk / 2         # Hohlkörperradius in cm

    print("\nGeometrie:\n----------------------")
    print("Stärke der oberen Deckschicht in cm: " +
          "{:02.1f}".format(d_OK,)+" cm")
    print("Achsabstand Hohlkörper in Querrichtung in cm: " +
          "{:02.1f}".format(e2,)+" cm")

    # Integrationsgrenzen
    # ----------------------
    a = 0                   # Obere Grenze
    b = r_hk + e/2          # Untere Grenze

    # Querschnittsfläche
    # ----------------------
    # Berechung des Querschnittes an der schwächsten Stelle
    A_brt, A_hk, A_net = crossSection(r_hk, h, D_hk, e)

    print("\nQuerschnittsfläche\n----------------------")
    print("A_net: "+"{:02.1f}".format(A_net,)+" cm^2")
    print("A_brt: "+"{:02.1f}".format(A_brt,)+" cm^2")
    print("Querschnittsreduktionsfaktor: "+"{:02.4f}".format(A_net/A_brt,))

    # Masse
    # ----------------------
    # Numerische Integration
    integral = quad(integrandCrossSection, a, b, args=(D_hk, e, h, d_OK))
    V_net = integral[0] * 2
    M_net = V_net * rho * 10 ** -6
    error_estim = integral[1]

    # Masse Bruttoquerschnitt
    V_brt = A_brt * (b-a) * 2
    M_brt = V_brt * rho * 10 ** -6

    print("\nMasse\n----------------------")
    print("M_net: "+"{:02.1f}".format(M_net,)+" kg")
    print("M_brt: "+"{:02.1f}".format(M_brt,)+" kg")
    print("Massensreduktionsfaktor: "+"{:02.4f}".format(M_net/M_brt,))

    # Flächenträgheitsmoment
    # ----------------------
    # Numerische Integration
    integral = quad(integrandMomentOfInertia, a, b, args=(D_hk, e, h, d_OK))
    I_s_net_int = integral[0]
    error_estim = integral[1]

    # Berechung des Mittelwertes über die betrachtete Länge
    I_s_net_m = I_s_net_int / (b - a)

    # Flächenträgheitsmoment Bruttoquerschnitt
    I_brt = (D_hk + e) * h ** 3 / 12

    print("\nFlächenträgheitsmoment\n----------------------")
    print("I_s_net_m: "+"{:02.1f}".format(I_s_net_m,)+" cm^4")
    print("I_brt: "+"{:02.1f}".format(I_brt,)+" cm^4")
    print("Steifigkeitsreduktionsfaktor: "+"{:02.4f}".format(I_s_net_m/I_brt,))


if __name__ == "__main__":
    main()
