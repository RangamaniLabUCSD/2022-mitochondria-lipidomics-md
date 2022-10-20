#!/usr/bin/env python

# =========================================================================
#
#  Module    : LOCAL STRESS FROM GROMACS TRAJECTORIES
#  File      : tensortools.py
#  Authors   : A. Torres-Sanchez and J. M. Vanegas
#  Modified  :
#  Purpose   : Compute the local stress from precomputed trajectories in GROMACS
#  Date      : 25/03/2015
#  Version   :
#  Changes   :
#
#     http://mdstress.org
#
#     This software is distributed WITHOUT ANY WARRANTY; without even
#     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#     PURPOSE.
#
#     Please, report any bug to either of us:
#     juan.m.vanegas@gmail.com
#     torres.sanchez.a@gmail.com
# =========================================================================
#
# References:
#
# Regarding this program:
# [1] Manual (parent folder/manual)
# [2] J. M. Vanegas; A. Torres-Sanchez; M. Arroyo; J. Chem. Theor. Comput. 10 (2), 691-702 (2014)
# [3] O. H. S. Ollila; H.J. Risselada; M. Louhivouri; E. Lindahl; I. Vattulainen; S.J. Marrink; Phys. Rev Lett. 102, 078101 (2009)
#
# General IKN framework and Central Force Decomposition
# [4] E. B. Tadmor; R. E. Miller; Modeling Materials: Continuum, Atomistic and Multiscale Techniques, Cambridge University Press (2011)
# [5] N. C. Admal; E. B. Tadmor; J. Elast. 100, 63 (2010)
#
# Covariant Central Force Decomposition
# [6] A. Torres-Sanchez; J. M. Vanegas; M. Arroyo; Submitted to PRL (2015)
# [7] A. Torres-Sanchez; J. M. Vanegas; M. Arroyo; In preparation (2015)
#
# Goetz-Lipowsky Decomposition
# [8] R. Goetz; R. Lipowsky; J. Chem. Phys. 108, 7397 (1998)
#


try:
    from LStensor import LStensor
except:
    print(
        "LStensor not in the current directory nor in python path. Add the location of your GROMACSLS bin folder to the PYTHONPATH (for instance type: 'export PYTHONPATH=$PYTHONPATH:/path/to/your/gromacsls/bin')"
    )
    exit(10)


import argparse
import struct as st
import os


def main():

    parser = argparse.ArgumentParser(
        description="Tensortools.py -- Swiss army knife of stress tensor and 3D density calculations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Juan M. Vanegas and Alejandro Torres-Sanchez
Please, report any bug to either of us:
juan.m.vanegas@gmail.com  and/or torres.sanchez.a@gmail.com""",
    )
    # INPUT
    parser.add_argument(
        "-f",
        nargs="+",
        required=True,
        help="Single or multiple binary input files (must all have the same grid size)",
        metavar="[input.bin]",
    )
    parser.add_argument(
        "-v", nargs="?", const=True, default=False, help="verbose", metavar=""
    )
    parser.add_argument(
        "--irep",
        nargs="?",
        default="spat_d",
        help="Input data representation, values: spat (spatial stress/density distributed on a grid, default), atom (stress per atom)",
        metavar="spat",
    )
    parser.add_argument(
        "--pdb",
        nargs="?",
        help="PDB file with atom coordinates (FOR STRESS PER ATOM ONLY)",
        metavar="[structure.pdb]",
    )
    parser.add_argument(
        "--itype",
        nargs="?",
        default="stress_d",
        help="Type of data such as density, stress tensor, or vector field: dens, stress (default), or vec",
        metavar="stress",
    )

    # PROCESSING DATA
    parser.add_argument(
        "--mif",
        nargs="?",
        default="avg",
        help="How to process multiple input files, values: avg (default), sum, or delta (subtract second to last input files from the first file)",
        metavar="avg",
    )
    parser.add_argument(
        "--sym",
        nargs="?",
        const=True,
        default=False,
        help="FOR STRESS ONLY. Separates the resultant stress in its symmetric and antisymmetric parts, and creates two outputs with the subscripts _sym and _asym respectively",
        metavar="",
    )
    parser.add_argument(
        "--prof",
        help="Output a profile along a given dimension, values: x, y, or z (default)",
        metavar="z",
    )
    parser.add_argument(
        "--integ",
        help="integrate out the stress profile along a given dimension, values: x, y, or z (default)",
        metavar="z",
    )
    parser.add_argument(
        "--gf",
        help="Process input through a gaussian filter of a given standard deviation (in units of the grid spacing, default=1.0)",
        metavar="1.0",
    )
    parser.add_argument(
        "--gridsp",
        nargs=3,
        help="(For atom to grid conversion) Grid spacing for atom 2 grid conversion (default = 0.1,0.1,0.1 in nm)",
        metavar=(0.1, 0.1, 0.1),
        default=(0.1, 0.1, 0.1),
    )
    parser.add_argument(
        "--trans",
        nargs=3,
        help="Translate data along a given axis (or multiple axes) by some integer number of grid units assuming periodic (wraparound) boundaries (default = 0,0,0)",
        metavar=(0, 0, 0),
        default=(0, 0, 0),
    )

    # OUTPUT
    parser.add_argument(
        "--orep",
        nargs="?",
        default="spat_d",
        help="Output data representation, values: spat (spatial stress/density distributed on a grid, default), atom (stress per atom)",
        metavar="spat",
    )
    parser.add_argument(
        "--oformat",
        nargs="?",
        default="bin_d",
        help="Output format, values: bin (i.e. binary .dat0, default), nc (NETCDF), txt (default when using --prof), chi (chimera attribute file), or pdb (stress per atom only, creates a separate file for each element in the tensor)",
        metavar="bin",
    )
    parser.add_argument(
        "-o", type=str, required=True, help="Output file", metavar="output.bin"
    )

    args = parser.parse_args()
    # print(args)

    # Load arguments
    inputfiles = args.f
    verbose = args.v

    # Tensortools input is always a binary file, so lets use the first integer bit to automagically determine the input file type: spatial stress tensor = 1, atom stress tensor = 2, and density = 3
    fp = open(inputfiles[0], "rb")
    dtype = st.unpack("i", fp.read(4))[0]

    if args.itype == "stress_d" and dtype == 3:
        itype = "dens"
    elif args.itype == "stress_d":
        itype = "stress"
    else:
        itype = args.itype

    if args.irep == "spat_d" and dtype == 2:
        irep = "atom"
    elif args.irep == "spat_d":
        irep = "spat"
    else:
        irep = args.irep

    if args.orep == "spat_d" and irep == "atom":
        orep = "atom"
    elif args.orep == "spat_d":
        orep = "spat"
    else:
        orep = args.orep

    outputfile = args.o
    basename, oext = os.path.splitext(outputfile)
    if (
        oext == ".bin" or oext == ".dat" or oext == ".dat0"
    ) and args.oformat == "bin_d":
        oformat = "bin"
    elif (oext == ".txt" or oext == ".xvg") and args.oformat == "bin_d":
        oformat = "txt"
    elif (oext == ".chi") and args.oformat == "bin_d":
        oformat = "chi"
    elif oext == ".nc" and args.oformat == "bin_d":
        oformat = "nc"
    elif oext == ".npy" and args.oformat == "bin_d":
        oformat = "npy"
    elif oext == "bin_d":
        oformat = "bin"
    else:
        oformat = args.oformat

    struct = args.pdb
    gridsp = [float(x) for x in args.gridsp]
    ax = [0, 0, 0]
    shift = [int(x) for x in args.trans]
    for i in (0, 1, 2):
        if shift[i] != 0:
            ax[i] = 1
    prof = args.prof
    integ = args.integ
    sym = args.sym

    # Check input data

    if check_args(irep, orep, oformat, itype, parser):
        return 1

    # Gaussian filter?
    if args.gf != None:
        bGF = True
        gf_sigma = float(args.gf)
    else:
        bGF = None

    # Average, sum, or delta?
    bAvg = args.mif

    # Type of data: density (order=0), vector (order=1), or stress (order=2)
    if itype == "dens":
        order = 0
    elif itype == "vec":
        order = 1
    elif itype == "stress":
        order = 2

    field = LStensor(order)

    field.verbose = verbose

    if irep == "spat":
        field.g_loaddata(inputfiles, bAvg)
        if ax[0] == 1 or ax[1] == 1 or ax[2] == 1:
            field.g_translate(ax, shift)

    if irep == "atom":
        field.a_loaddata(inputfiles, bAvg)

    if struct != None:
        field.a_loadstr(struct)
        if orep == "spat":
            field.a_2grid(gridsp)

    if prof != None:
        field.g_prof(prof)
        # oformat = "txt"
        # oformat = "npy"

    if integ != None:
        field.g_intout(integ)

    # Apply Gaussian filter if necessary and overwrite data with it
    if bGF == True:
        field.g_gfilt(gf_sigma)

    # If sym options is active separate the symmetric and antisymmetric components (ONLY FOR STRESS)
    if sym:
        field.g_symasym()

    # Write data to file
    if orep == "spat":
        if not (sym):
            if oformat == "bin":
                field.g_savebin(outputfile)
            elif oformat == "nc":
                field.g_savenc(outputfile)
            elif oformat == "txt":
                field.g_savetxt(outputfile)
            elif oformat == "npy":
                field.g_savenpy(outputfile)
        else:
            outname, outext = outputfile.split(".")
            namesym = outname + "_sym." + outext
            nameasym = outname + "_asym." + outext
            if oformat == "bin":
                field.sym.g_savebin(namesym)
                field.asym.g_savebin(nameasym)
            elif oformat == "nc":
                field.sym.g_savenc(namesym)
                field.asym.g_savenc(nameasym)
            elif oformat == "txt":
                field.sym.g_savetxt(namesym)
                field.asym.g_savetxt(nameasym)
    else:
        if oformat == "bin":
            field.a_savebin(outputfile)
        elif oformat == "pdb":
            field.a_savepdb(outputfile)
        elif oformat == "txt":
            field.a_savetxt(outputfile)
        elif oformat == "chi":
            field.a_savechiattr(outputfile)

    return 0


def check_args(irep, orep, oformat, itype, parser):

    args = parser.parse_args()

    if irep == "spat" and args.pdb != None:
        print(
            "INPUT ERROR: pdb files are only allowed for stress per atom but irep is set to 'spat'."
        )
        parser.print_help()
        return 1

    if (
        oformat != "bin"
        and oformat != "nc"
        and oformat != "txt"
        and oformat != "pdb"
        and oformat != "chi"
        and oformat != "npy"
    ):
        print(
            "INPUT ERROR: Output type must be either 'bin' or 'nc' or 'txt' or 'pdb' or 'npy'."
        )
        parser.print_help()
        return 1

    elif (oformat == "nc") and orep == "atom":
        print("INPUT ERROR: Output type cannot be 'nc' when orep is 'atom")
        parser.print_help()
        return 1

    if args.mif != "avg" and args.mif != "sum" and args.mif != "delta":
        print(
            "INPUT ERROR: Option for processing must be either avg (average), sum, or delta."
        )
        parser.print_help()
        return 1

    if args.sym == True:
        if itype != "stress":
            print("INPUT ERROR: The data type must be stress with the --sym option!")
            parser.print_help()
            return 1

    if args.prof != None:
        if args.prof != "x" and args.prof != "y" and args.prof != "z":
            print("INPUT ERROR: The profile option is not 'x' 'y' or 'z'!")
            parser.print_help()
            return 1
        if args.integ != None:
            print("INPUT ERROR: The options prof and integ are incompatible!")
            parser.print_help()
            return 1

    if args.integ != None:
        if args.integ != "x" and args.integ != "y" and args.integ != "z":
            print("INPUT ERROR: The integration option is not 'x' 'y' or 'z'!")
            parser.print_help()
            return 1

    if oformat == "pdb":
        if orep != "atom":
            print(
                "INPUT ERROR: PDB output is only compatible with atom output representation"
            )
            parser.print_help()
            return 1

    if orep == "atom":
        if irep == "spat":
            print(
                "INPUT ERROR: Not conversion from grid data to atomic data is possible"
            )
            parser.print_help()
            return 1
        if args.integ != None:
            print(
                "INPUT ERROR: Integration does not make sense for atomic data (orep == atom)"
            )
            parser.print_help()
            return 1
        if args.prof != None:
            print(
                "INPUT ERROR: Profile does not make sense for atomic data (orep == atom)"
            )
            parser.print_help()
            return 1
        if args.gf != None:
            print(
                "INPUT ERROR: gaussian filter does not make sense for atomic data (orep == atom)"
            )
            parser.print_help()
            return 1
        if args.sym:
            print("INPUT ERROR: sym not valid for atomic data (orep == atom)")
            parser.print_help()
            return 1
    return 0


if __name__ == "__main__":
    main()
