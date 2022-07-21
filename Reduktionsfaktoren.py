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


def shearCrossSection(d_s, D_hk, r_hk, d_OK, e):
    """Wirksame Schubfläche am betrachteten Schnitt entlang x-Achse
    :param d_s: Statische Nutzhöhe in cm
    :param D_hk: Hohlkörperdurchmesser in cm
    :param r_hk: Hohlkörperradius in cm
    :param d_OK: obere Deckschicht in cm
    :param e: Abstand Hohlkörper in Querrichtung in cm
    :rtype A_brt_45: Bruttoquerschnitt im 45° Schnitt in cm^2
    :rtype A_net_45: Nettoquerschnitt  im 45° Schnitt in cm^2
    """
    A_brt_45 = np.sqrt(2) * d_s * (D_hk + e)

    # Für Definition der Geometrie siehe validierung/wirksame_Schubfläche.dwg
    # Fall 1: Schwerachse Bewehrung unterhalb Schnittpunkt 45° Schnitt/Hohlkörper
    if (d_OK + r_hk + np.sqrt(2)/2 * r_hk) <= d_s:
        A_hk = np.pi * (D_hk / 2) ** 2

    # Fall 2: Schwerachse Bewehrung oberhalb Schnittpunkt 45° Schnitt/Hohlkörper
    elif d_OK + r_hk < d_s:
        # Fläche des Hohlkörpers
        A_hk = np.pi * (D_hk / 2) ** 2
        # Abstand Schwerpunkt Hohlkörper zu Schwerachse Bewehrung
        d_s_hk = d_s - d_OK - D_hk / 2
        # Abstand Schwerpunkt Hohlkörper zu Schwerachse Bewehrung im 45° Schnitt
        d_s_hk_45 = d_s_hk / np.cos(np.pi / 4)
        # Höhe Ausschnitt Kreissegment unterhalb Schwerachse Bewehrung
        h_ks_45 = (D_hk / 2) - d_s_hk_45
        # Öffnungswinkel Kreissegment unterhalb Schwerachse Bewehrung
        alpha = 2 * np.arccos(1 - h_ks_45 / (D_hk / 2))
        # Fläche Kreissegment unterhalb Schwerachse Bewehrung
        A_ks = (D_hk / 2) ** 2 / 2 * (alpha - np.sin(alpha))
        # Querschnitt Hohlkörper oberhalb Schwerachse Bewehrung
        A_hk = A_hk - A_ks

    else:
        print("Schwerachse untere Bewehrung liegt oberhalb Schwerachse Hohlkörper. \n")
        print("Bitte Hohlkörperlayout prüfen. \n")

    A_net_45 = A_brt_45 - A_hk

    return A_brt_45, A_net_45


def centerOfGravity(h, D_hk, d_OK, A_brt, A_hk, A_net):
    """Schwerpunkte am betrachteten Schnitt entlang x-Achse
    :param h: Deckenstärke in cm
    :param D_hk: Hohlkörperdurchmesser in cm
    :param d_OK: obere Deckschicht in cm
    :param A_brt: Bruttoquerschnitt in cm^2
    :param A_hk: Hohlkörperquerschnitt in cm^2
    :param A_net: Nettoquerschnitt in cm^2
    :rtype zs_brt: Schwerpunkt des Bruttoquerschnitts in cm
    :rtype zs_hk: Schwerpunkt des Hohlkörperquerschnitts in cm
    :rtype zs_net: Schwerpunkt des Nettoquerschnitts in cm
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
    :param zs_brt: Schwerpunkt des Bruttoquerschnitts in cm
    :param zs_hk: Schwerpunkt des Hohlkörperquerschnitts in cm
    :param zs_net: Schwerpunkt des Nettoquerschnitts in cm
    :rtype I_brt: Flächenträgheitsmoment des Bruttoquerschnitt in cm^4
    :rtype I_s_net: Flächenträgheitsmoment des Nettoquerschnittes in cm^4
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

    return A_net


def integrandCenterOfGravity(x, D_hk, e, h, d_OK):
    # Radius
    r_x = radiusAtSection(D_hk, x, e)

    # Querschnittsflächen
    A_brt, A_hk, A_net = crossSection(r_x, h, D_hk, e)

    # Schwerpunkte
    zs_brt, zs_hk, zs_net = centerOfGravity(
        h, D_hk, d_OK, A_brt, A_hk, A_net)

    return zs_net


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

    return I_s_net


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
    print("\nEINGABEPARAMETER\n----------------------")
    h = getValidInput("Deckenstärke in cm: ", 0)
    D_hk = getValidInput("Hohlkörperdurchmesser in cm: ", 0, h)
    d_UK = getValidInput("Stärke der unteren Deckschicht in cm: ", 0, h-D_hk)
    e = getValidInput("Hohlkörperabstand in Querrichtung in cm: ", 0)
    rho = getValidInput("Rohdichte Beton in kg/m³: ", 0)
    d_s = getValidInput("Statische Nutzhöhe in cm: ", 0)

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

    print("\nQUERSCHNITTSWERTE AM SCHWÄCHSTEN QUERSCHNITT DER EINHEITSZELLE\n----------------------")

    # Querschnittsfläche
    # ----------------------
    # Berechung des Querschnittes an der schwächsten Stelle
    A_brt, A_hk, A_net = crossSection(r_hk, h, D_hk, e)

    print("Querschnittsfläche\n-----------")
    print("A_net: "+"{:02.1f}".format(A_net,)+" cm^2")
    print("A_brt: "+"{:02.1f}".format(A_brt,)+" cm^2")
    print("Querschnittsreduktionsfaktor: "+"{:02.4f}".format(A_net/A_brt,))

    # Wirksame Schubfläche
    # ----------------------
    # Berechung des Querschnittes an der schwächsten Stelle
    A_brt_45, A_net_45 = shearCrossSection(d_s, D_hk, r_hk, d_OK, e)

    print("\nWirksame Schubfläche\n-----------")
    print("A_net_45: "+"{:02.1f}".format(A_net_45,)+" cm^2")
    print("A_brt_45: "+"{:02.1f}".format(A_brt_45,)+" cm^2")
    print("Reduktionsfaktor Wirksame Schubfläche: " +
          "{:02.4f}".format(A_net_45/A_brt_45,))

    # Schwerpunkt
    # ----------------------
    # Berechung des Schwerpunkts an der schwächsten Stelle
    zs_brt, zs_hk, zs_net = centerOfGravity(h, D_hk, d_OK, A_brt, A_hk, A_net)

    print("\nQuerschnittsschwerpunkt ab Oberkante\n-----------")
    print("zs_net: "+"{:02.1f}".format(zs_net,)+" cm")

    # Flächenträgheitsmoment
    # ----------------------
    # Berechung des Flächenträgheitsmoments an der schwächsten Stelle
    I_brt, I_s_net = momentOfInertia(
        r_hk, h, D_hk, e, A_brt, A_hk, zs_brt, zs_hk, zs_net)

    print("\nFlächenträgheitsmoment\n-----------")
    print("I_s_net: "+"{:02.1f}".format(I_s_net,)+" cm^4")
    print("I_brt: "+"{:02.1f}".format(I_brt,)+" cm^4")

    print("\nGEMITTELTE QUERSCHNITTSWERTE DER EINHEITSZELLE\n----------------------")

    # Masse
    # ----------------------
    # Numerische Integration zur Mittlung der Masse
    integral = quad(integrandCrossSection, a, b, args=(D_hk, e, h, d_OK))
    V_net = integral[0] * 2
    M_net = V_net * rho * 10 ** -6
    error_estim = integral[1]

    # Masse Bruttoquerschnitt
    V_brt = A_brt * (b-a) * 2
    M_brt = V_brt * rho * 10 ** -6

    print("Masse\n-----------")
    print("M_net: "+"{:02.1f}".format(M_net,)+" kg")
    print("M_brt: "+"{:02.1f}".format(M_brt,)+" kg")
    print("Massensreduktionsfaktor: "+"{:02.4f}".format(M_net/M_brt,))

    # Schwerpunkte
    # ----------------------
    # Numerische Integration zur Mittlung der Schwerpunkte
    integral = quad(integrandCenterOfGravity, a, b, args=(D_hk, e, h, d_OK))
    zs_net_int = integral[0]
    error_estim = integral[1]

    # Berechung des Mittelwertes über die betrachtete Länge
    zs_net_m = zs_net_int / (b - a)

    print("\nQuerschnittsschwerpunkt ab Oberkante\n-----------")
    print("zs_net_m: "+"{:02.2f}".format(zs_net_m,)+" cm")

    # Flächenträgheitsmoment
    # ----------------------
    # Numerische Integration zur Mittlung des FLÄCHENTRÄGHEITSMOMENTS
    integral = quad(integrandMomentOfInertia, a, b, args=(D_hk, e, h, d_OK))
    I_s_net_int = integral[0]
    error_estim = integral[1]

    # Berechung des Mittelwertes über die betrachtete Länge
    I_s_net_m = I_s_net_int / (b - a)

    # Flächenträgheitsmoment Bruttoquerschnitt
    I_brt = (D_hk + e) * h ** 3 / 12

    print("\nFlächenträgheitsmoment\n-----------")
    print("I_s_net_m: "+"{:02.1f}".format(I_s_net_m,)+" cm^4")
    print("I_brt: "+"{:02.1f}".format(I_brt,)+" cm^4")
    print("Steifigkeitsreduktionsfaktor: " +
          "{:02.4f}".format(I_s_net_m/I_brt,)+"\n")


if __name__ == "__main__":
    main()
