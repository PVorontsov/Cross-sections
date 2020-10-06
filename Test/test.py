#import tensorflow
import random
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
from sklearn.linear_model import LinearRegression

# line 1 points. Calculated in BOLSIG for the initial set of cross sections

# Vector with reduced electric field (Td) values
x1 = [
      3.070000e-1,
      3.620000e-1,
      4.410000e-1,
      5.430000e-1,
      6.850000e-1,
      8.110000e-1,
      9.760000e-1,
      1.220000e+0,
      1.519000e+0,
      1.763000e+0,
      1.991000e+0,
      2.274000e+0,
      2.526000e+0,
      2.825000e+0,
      3.080000e+0,
      3.330000e+0,
      3.630000e+0,
      3.790000e+0,
      4.060000e+0,
      4.440000e+0,
      4.820000e+0,
      5.190000e+0,
      5.410000e+0,
      5.630000e+0,
      7.436000e+0,
      9.964000e+0,
      1.248000e+1,
      1.502000e+1,
      1.754000e+1,
      2.007000e+1,
      2.260000e+1,
      2.512000e+1,
      2.764000e+1,
      3.017000e+1,
      3.269000e+1,
      3.526000e+1,
      3.788000e+1,
      4.042000e+1]


# Vector with experimental mobility values.
y1 = [
      8.827400e+24,
      7.624300e+24,
      6.371900e+24,
      5.322300e+24,
      4.321200e+24,
      3.699100e+24,
      3.176200e+24,
      2.704900e+24,
      2.304100e+24,
      2.042000e+24,
      1.908600e+24,
      1.715000e+24,
      1.543900e+24,
      1.415900e+24,
      1.298700e+24,
      1.231200e+24,
      1.101900e+24,
      1.081800e+24,
      1.059100e+24,
      1.036000e+24,
      1.016600e+24,
      1.001900e+24,
      9.981500e+23,
      1.012400e+24,
      9.773000e+23,
      9.763000e+23,
      9.687000e+23,
      9.584000e+23,
      9.494000e+23,
      9.366000e+23,
      9.247000e+23,
      9.146000e+23,
      9.053000e+23,
      8.957000e+23,
      8.863000e+23,
      8.780000e+23,
      8.679000e+23,
      8.578000e+23]

y2 = [
      8.827400e+24,
      7.624300e+24,
      6.371900e+24,
      5.322300e+24,
      4.321200e+24,
      3.699100e+24,
      3.176200e+24,
      2.704900e+24,
      2.304100e+24,
      2.042000e+24,
      1.908600e+24,
      1.715000e+24,
      1.543900e+24,
      1.415900e+24,
      1.298700e+24,
      1.231200e+24,
      1.101900e+24,
      1.081800e+24,
      1.059100e+24,
      1.036000e+24,
      1.016600e+24,
      1.001900e+24,
      9.981500e+23,
      1.012400e+24,
      9.773000e+23,
      9.763000e+23,
      9.687000e+23,
      9.584000e+23,
      9.494000e+23,
      9.366000e+23,
      9.247000e+23,
      9.146000e+23,
      9.053000e+23,
      8.957000e+23,
      8.863000e+23,
      8.780000e+23,
      8.679000e+23,
      8.578000e+23]

plt.plot(x1, y1)
plt.ylabel('Graph')
plt.show()

# initial cross sections

# np array for effective. Energy and cross sections.
effective = np.array(
      [

      [
      1.000000e-3,
      2.000000e-3,
      3.000000e-3,
      5.000000e-3,
      7.000000e-3,
      8.500000e-3,
      1.000000e-2,
      1.500000e-2,
      2.000000e-2,
      3.000000e-2,
      4.000000e-2,
      5.000000e-2,
      7.000000e-2,
      1.000000e-1,
      1.200000e-1,
      1.500000e-1,
      1.700000e-1,
      2.000000e-1,
      2.500000e-1,
      3.000000e-1,
      3.500000e-1,
      4.000000e-1,
      5.000000e-1,
      7.000000e-1,
      1.000000e+0,
      1.200000e+0,
      1.300000e+0,
      1.500000e+0,
      1.700000e+0,
      1.900000e+0,
      2.100000e+0,
      2.200000e+0,
      2.500000e+0,
      2.800000e+0,
      3.000000e+0,
      3.300000e+0,
      3.600000e+0,
      4.000000e+0,
      4.500000e+0,
      5.000000e+0,
      6.000000e+0,
      7.000000e+0,
      8.000000e+0,
      1.000000e+1,
      1.200000e+1,
      1.500000e+1,
      1.700000e+1,
      2.000000e+1,
      2.500000e+1,
      3.000000e+1,
      5.000000e+1,
      7.500000e+1,
      1.000000e+2,
      1.500000e+2,
      2.000000e+2,
      3.000000e+2,
      5.000000e+2,
      7.000000e+2,
      1.000000e+3,
      1.500000e+3,
      2.000000e+3,
      3.000000e+3,
      5.000000e+3,
      7.000000e+3,
      1.000000e+4
      ],

      [
      7.500000e-20,
      7.100000e-20,
      6.700000e-20,
      6.100000e-20,
      5.400000e-20,
      5.050000e-20,
      4.600000e-20,
      3.750000e-20,
      3.250000e-20,
      2.500000e-20,
      2.050000e-20,
      1.730000e-20,
      1.130000e-20,
      5.900000e-21,
      4.000000e-21,
      2.300000e-21,
      1.600000e-21,
      1.030000e-21,
      9.100000e-22,
      1.530000e-21,
      2.350000e-21,
      3.300000e-21,
      5.100000e-21,
      8.600000e-21,
      1.380000e-20,
      1.660000e-20,
      1.820000e-20,
      2.100000e-20,
      2.300000e-20,
      2.500000e-20,
      2.800000e-20,
      2.900000e-20,
      3.300000e-20,
      3.800000e-20,
      4.100000e-20,
      4.500000e-20,
      4.900000e-20,
      5.400000e-20,
      6.100000e-20,
      6.700000e-20,
      8.100000e-20,
      9.600000e-20,
      1.170000e-19,
      1.500000e-19,
      1.520000e-19,
      1.410000e-19,
      1.310000e-19,
      1.100000e-19,
      9.450000e-20,
      8.740000e-20,
      6.900000e-20,
      5.850000e-20,
      5.250000e-20,
      4.240000e-20,
      3.760000e-20,
      3.020000e-20,
      2.100000e-20,
      1.640000e-20,
      1.210000e-20,
      8.800000e-21,
      6.600000e-21,
      4.500000e-21,
      3.100000e-21,
      2.300000e-21,
      1.750000e-21
      ]
      
      ])

# np array for excitation. Energy and cross sections.
excitation = np.array(
      [

      [
      1.150000e+1,
      1.270000e+1,
      1.370000e+1,
      1.470000e+1,
      1.590000e+1,
      1.650000e+1,
      1.750000e+1,
      1.850000e+1,
      1.990000e+1,
      2.220000e+1,
      2.470000e+1,
      2.700000e+1,
      3.000000e+1,
      3.300000e+1,
      3.530000e+1,
      4.200000e+1,
      4.800000e+1,
      5.200000e+1,
      7.000000e+1,
      1.000000e+2,
      1.500000e+2,
      2.000000e+2,
      3.000000e+2,
      5.000000e+2,
      7.000000e+2,
      1.000000e+3,
      1.500000e+3,
      2.000000e+3,
      3.000000e+3,
      5.000000e+3,
      7.000000e+3,
      1.000000e+4
      ],

      [
      0.000000e+0,
      7.000000e-22,
      1.410000e-21,
      2.280000e-21,
      3.800000e-21,
      4.800000e-21,
      6.100000e-21,
      7.500000e-21,
      9.200000e-21,
      1.170000e-20,
      1.330000e-20,
      1.420000e-20,
      1.440000e-20,
      1.410000e-20,
      1.340000e-20,
      1.250000e-20,
      1.160000e-20,
      1.110000e-20,
      9.400000e-21,
      7.600000e-21,
      6.000000e-21,
      5.050000e-21,
      3.950000e-21,
      2.800000e-21,
      2.250000e-21,
      1.770000e-21,
      1.360000e-21,
      1.100000e-21,
      8.300000e-22,
      5.800000e-22,
      4.500000e-22,
      3.500000e-22
      ]

      ])

# np array for ionization. Energy and cross sections.
ionization = np.array(
      [

      [
      1.580000e+1,
      1.600000e+1,
      1.700000e+1,
      1.800000e+1,
      2.000000e+1,
      2.200000e+1,
      2.375000e+1,
      2.500000e+1,
      2.650000e+1,
      3.000000e+1,
      3.250000e+1,
      3.500000e+1,
      3.750000e+1,
      4.000000e+1,
      5.000000e+1,
      5.500000e+1,
      1.000000e+2,
      1.500000e+2,
      2.000000e+2,
      3.000000e+2,
      5.000000e+2,
      7.000000e+2,
      1.000000e+3,
      1.500000e+3,
      2.000000e+3,
      3.000000e+3,
      5.000000e+3,
      7.000000e+3,
      1.000000e+4 
      ],

      [
      0.000000e+0,
      2.020000e-22,
      1.340000e-21,
      2.940000e-21,
      6.300000e-21,
      9.300000e-21,
      1.150000e-20,
      1.300000e-20,
      1.450000e-20,
      1.800000e-20,
      1.990000e-20,
      2.170000e-20,
      2.310000e-20,
      2.390000e-20,
      2.530000e-20,
      2.600000e-20,
      2.850000e-20,
      2.520000e-20,
      2.390000e-20,
      2.000000e-20,
      1.450000e-20,
      1.150000e-20,
      8.600000e-21,
      6.400000e-21,
      5.200000e-21,
      3.600000e-21,
      2.400000e-21,
      1.800000e-21,
      1.350000e-21
      ]

      ])

# function to create a cross section file and a bolsig file in required format.

#s1 and s2 are strings
def createfiles(s1, s2):

      # creating file with cross sections data

      f = open(s1, "a")
      f.write("EFFECTIVE\n")
      f.write("Ar\n")
      f.write(" 1.360000e-5\n")
      f.write("-----------------------------\n")

      #print(effective.shape[1])

      # looping through EFFECTIVE
      for x in range(0, effective.shape[1]):

          # writing energy (eV) and cross section (m2) values into a file
          f.write("{:.6e}".format(effective[0, x]))
          f.write("      ")
          f.write("{:.6e}".format(effective[1, x]))
          f.write("\n")

      f.write("-----------------------------\n")
      f.write("EXCITATION\n")
      f.write("Ar -> Ar*(11.5eV)\n")
      f.write(" 1.150000e+1\n")
      f.write("-----------------------------\n")

      # looping through EXCITATION
      for x in range(0, excitation.shape[1]):

          # writing energy (eV) and cross section (m2) values into a file
          f.write("{:.6e}".format(excitation[0, x]))
          f.write("      ")
          f.write("{:.6e}".format(excitation[1, x]))
          f.write("\n")

      f.write("-----------------------------\n")
      f.write("IONIZATION\n")
      f.write("Ar -> Ar^+\n")
      f.write(" 1.580000e+1\n")
      f.write("-----------------------------\n")

      # looping through IONIZATION
      for x in range(0, ionization.shape[1]):

          # writing energy (eV) and cross section (m2) values into a file
          f.write("{:.6e}".format(ionization[0, x]))
          f.write("      ")
          f.write("{:.6e}".format(ionization[1, x]))
          f.write("\n")

      f.write("-----------------------------\n")

      # closing the file
      f.close()

      # creating a bolsig file
      f = open(s2, "a")
      f.write("READCOLLISIONS\n")

      #using cross sections data from s1
      f.write(s1)
      f.write("\n")
      f.write("Ar\n")
      f.write("1\n\n")
      f.write("CONDITIONS\n")
      f.write("VAR\n")
      f.write("0.\n0.\n300.\n300.\n0.\n0.\n1e18\n1.\n1.\n1\n1\n1\n0.\n200\n0\n"
            "200.\n1e-10\n1e-4\n1000\n0.99 0.01\n1\n\nRUN\n")

      # reduced electric field (Td) values to input to BOLSIG
      f.write("3.07e-1\n3.62e-1\n4.41e-1\n5.43e-1\n6.85e-1\n8.11e-1\n9.76e-1\n1.22e+0\n1.519e+0\n"
              "1.763e+0\n1.991e+0\n2.274e+0\n2.526e+0\n2.825e+0\n3.08e+0\n3.33e+0\n3.63e+0\n3.79e+0\n"
              "4.06e+0\n4.44e+0\n4.82e+0\n5.19e+0\n5.41e+0\n5.63e+0\n7.436e+0\n9.964e+0\n1.248e+1\n"
              "1.502e+1\n1.754e+1\n2.007e+1\n2.26e+1\n2.512e+1\n2.764e+1\n3.017e+1\n3.269e+1\n3.526e+1\n"
              "3.788e+1\n4.042e+1\n\n")
      f.write("SAVERESULTS\n")

      # producing output file name
      f.write("output" + s2 + "\n")
      f.write("3\n0\n1\n0\n0\n0\n0\n0")

      f.close()


# creating multiple cross section and bolsig files for training data

# npvector_with_errors = np.zeros(25, 4)

# number of training examples
number_tr_ex = 40

# np vector with coefficicents for cross sections. EFFECTIVE, IONIZATION, and EXCITATION.
# Initially filled with zeros
X = np.zeros((number_tr_ex, 3))

for p in X:
    print(p)

vector_with_errors = [0] * number_tr_ex

# coefficients used to modify EFFECTIVE part of the cross sections file.
vector_with_eff_coefs = [0] * number_tr_ex

# coefficients used to modify EXCITATION part of the cross sections file.
vector_with_exci_coefs = [0] * number_tr_ex

# #coefficients used to modify IONIZATION part of the cross sections file.
vector_with_ion_coefs = [0] * number_tr_ex

#initiall coefficients are zero. We are checking error for the existing cross sections file.

eff_coef = 1

exci_coef = 1

ion_coef = 1

for j in range(0, number_tr_ex):

    X[j, 0] = eff_coef

    vector_with_eff_coefs[j] = eff_coef
    print(vector_with_eff_coefs[j])

    X[j, 1] = exci_coef

    vector_with_exci_coefs[j] = exci_coef
    print(vector_with_exci_coefs[j])

    X[j, 2] = ion_coef

    vector_with_ion_coefs[j] = ion_coef
    print(vector_with_ion_coefs[j])

    # str is a python function to convert integer to string

    eff_coef = 1

    exci_coef = 1

    ion_coef = 1

    str1 = "myxsecfile" + str(j+1) + ".txt"
    str2 = "mybolsigfile" + str(j+1) + ".txt"

    #creating cross sections and BOLSIG files
    createfiles(str1, str2)
    
    # launching BOLSIG- with str2 as a command-line argument
    subprocess.run(['bolsigminus.exe', str2])

    # deleting created cross-sections and BOLSIG files, since we do not need them anymore.
    os.remove(str1)
    os.remove(str2)

    # this is where the output for the current cross sections set is stored
    outputfilename = "output" + str2

    #
    a_file = open(outputfilename, "r")

    # creating the "lines" list. Elements of this list are lines from the output file.
    lines = a_file.readlines()
    a_file.close()

    # deleting first 49 elements of the list, because we are only interested in Mobility data.
    for i in range(0, 49):
          del lines[0]
    # deleting elements of the list after Mobility data.
    for i in range(38, len(lines)):
          del lines[38]

    #mobilityvector = [0] * len(y1)

    sum_of_squared_differences = 0.0
    i = 0
    for line in lines:

        # We don't need reduced electric field values anymore, therefore slicing each
        # string that is an element of the "lines" list
        line = (line[len(line) - 11: len(line) - 1])
        #print(line)

        # float is a function to convert String into float number
        mobility = float(line)
        # updating the sum of square differences
        sum_of_squared_differences = sum_of_squared_differences + ((mobility - y2[i])**2)

        y1[i] = mobility

        #mobilityvector[i] = mobility
        i = i + 1
        #print(mobility)
        #print(sum_of_squared_differences)

    #vector_with_errors[4, j] = sum_of_squared_differences

    #npvector_with_errors[:, 1, 1, 1] =

    vector_with_errors[j] = sum_of_squared_differences

    # removing files that we do not need anymore
    os.remove(outputfilename)
    os.remove('bolsiglog.txt')

    plt.plot(x1, y1)
    plt.ylabel('Graph')
    plt.show()

    # updating coefficients
    eff_coef *= random.uniform(0.5, 1.5)
    #eff_coef *= 0.89218444
    exci_coef *= random.uniform(0.5, 1.5)
    #exci_coef *= 0.77861771
    ion_coef *= random.uniform(0.5, 1.5)
    #ion_coef *= 1.6405943
    # updating cross sections data

    effective = np.array(
          [

                [
                      1.000000e-3,
                      2.000000e-3,
                      3.000000e-3,
                      5.000000e-3,
                      7.000000e-3,
                      8.500000e-3,
                      1.000000e-2,
                      1.500000e-2,
                      2.000000e-2,
                      3.000000e-2,
                      4.000000e-2,
                      5.000000e-2,
                      7.000000e-2,
                      1.000000e-1,
                      1.200000e-1,
                      1.500000e-1,
                      1.700000e-1,
                      2.000000e-1,
                      2.500000e-1,
                      3.000000e-1,
                      3.500000e-1,
                      4.000000e-1,
                      5.000000e-1,
                      7.000000e-1,
                      1.000000e+0,
                      1.200000e+0,
                      1.300000e+0,
                      1.500000e+0,
                      1.700000e+0,
                      1.900000e+0,
                      2.100000e+0,
                      2.200000e+0,
                      2.500000e+0,
                      2.800000e+0,
                      3.000000e+0,
                      3.300000e+0,
                      3.600000e+0,
                      4.000000e+0,
                      4.500000e+0,
                      5.000000e+0,
                      6.000000e+0,
                      7.000000e+0,
                      8.000000e+0,
                      1.000000e+1,
                      1.200000e+1,
                      1.500000e+1,
                      1.700000e+1,
                      2.000000e+1,
                      2.500000e+1,
                      3.000000e+1,
                      5.000000e+1,
                      7.500000e+1,
                      1.000000e+2,
                      1.500000e+2,
                      2.000000e+2,
                      3.000000e+2,
                      5.000000e+2,
                      7.000000e+2,
                      1.000000e+3,
                      1.500000e+3,
                      2.000000e+3,
                      3.000000e+3,
                      5.000000e+3,
                      7.000000e+3,
                      1.000000e+4
                ],

                [
                      7.500000e-20,
                      7.100000e-20,
                      6.700000e-20,
                      6.100000e-20,
                      5.400000e-20,
                      5.050000e-20,
                      4.600000e-20,
                      3.750000e-20,
                      3.250000e-20,
                      2.500000e-20,
                      2.050000e-20,
                      1.730000e-20,
                      1.130000e-20,
                      5.900000e-21,
                      4.000000e-21,
                      2.300000e-21,
                      1.600000e-21,
                      1.030000e-21,
                      9.100000e-22,
                      1.530000e-21,
                      2.350000e-21,
                      3.300000e-21,
                      5.100000e-21,
                      8.600000e-21,
                      1.380000e-20,
                      1.660000e-20,
                      1.820000e-20,
                      2.100000e-20,
                      2.300000e-20,
                      2.500000e-20,
                      2.800000e-20,
                      2.900000e-20,
                      3.300000e-20,
                      3.800000e-20,
                      4.100000e-20,
                      4.500000e-20,
                      4.900000e-20,
                      5.400000e-20,
                      6.100000e-20,
                      6.700000e-20,
                      8.100000e-20,
                      9.600000e-20,
                      1.170000e-19,
                      1.500000e-19,
                      1.520000e-19,
                      1.410000e-19,
                      1.310000e-19,
                      1.100000e-19,
                      9.450000e-20,
                      8.740000e-20,
                      6.900000e-20,
                      5.850000e-20,
                      5.250000e-20,
                      4.240000e-20,
                      3.760000e-20,
                      3.020000e-20,
                      2.100000e-20,
                      1.640000e-20,
                      1.210000e-20,
                      8.800000e-21,
                      6.600000e-21,
                      4.500000e-21,
                      3.100000e-21,
                      2.300000e-21,
                      1.750000e-21
                ]

          ])

    # np array for excitation. Energy and cross sections.
    excitation = np.array(
          [

                [
                      1.150000e+1,
                      1.270000e+1,
                      1.370000e+1,
                      1.470000e+1,
                      1.590000e+1,
                      1.650000e+1,
                      1.750000e+1,
                      1.850000e+1,
                      1.990000e+1,
                      2.220000e+1,
                      2.470000e+1,
                      2.700000e+1,
                      3.000000e+1,
                      3.300000e+1,
                      3.530000e+1,
                      4.200000e+1,
                      4.800000e+1,
                      5.200000e+1,
                      7.000000e+1,
                      1.000000e+2,
                      1.500000e+2,
                      2.000000e+2,
                      3.000000e+2,
                      5.000000e+2,
                      7.000000e+2,
                      1.000000e+3,
                      1.500000e+3,
                      2.000000e+3,
                      3.000000e+3,
                      5.000000e+3,
                      7.000000e+3,
                      1.000000e+4
                ],

                [
                      0.000000e+0,
                      7.000000e-22,
                      1.410000e-21,
                      2.280000e-21,
                      3.800000e-21,
                      4.800000e-21,
                      6.100000e-21,
                      7.500000e-21,
                      9.200000e-21,
                      1.170000e-20,
                      1.330000e-20,
                      1.420000e-20,
                      1.440000e-20,
                      1.410000e-20,
                      1.340000e-20,
                      1.250000e-20,
                      1.160000e-20,
                      1.110000e-20,
                      9.400000e-21,
                      7.600000e-21,
                      6.000000e-21,
                      5.050000e-21,
                      3.950000e-21,
                      2.800000e-21,
                      2.250000e-21,
                      1.770000e-21,
                      1.360000e-21,
                      1.100000e-21,
                      8.300000e-22,
                      5.800000e-22,
                      4.500000e-22,
                      3.500000e-22
                ]

          ])

    # np array for ionization. Energy and cross sections.
    ionization = np.array(
          [

                [
                      1.580000e+1,
                      1.600000e+1,
                      1.700000e+1,
                      1.800000e+1,
                      2.000000e+1,
                      2.200000e+1,
                      2.375000e+1,
                      2.500000e+1,
                      2.650000e+1,
                      3.000000e+1,
                      3.250000e+1,
                      3.500000e+1,
                      3.750000e+1,
                      4.000000e+1,
                      5.000000e+1,
                      5.500000e+1,
                      1.000000e+2,
                      1.500000e+2,
                      2.000000e+2,
                      3.000000e+2,
                      5.000000e+2,
                      7.000000e+2,
                      1.000000e+3,
                      1.500000e+3,
                      2.000000e+3,
                      3.000000e+3,
                      5.000000e+3,
                      7.000000e+3,
                      1.000000e+4
                ],

                [
                      0.000000e+0,
                      2.020000e-22,
                      1.340000e-21,
                      2.940000e-21,
                      6.300000e-21,
                      9.300000e-21,
                      1.150000e-20,
                      1.300000e-20,
                      1.450000e-20,
                      1.800000e-20,
                      1.990000e-20,
                      2.170000e-20,
                      2.310000e-20,
                      2.390000e-20,
                      2.530000e-20,
                      2.600000e-20,
                      2.850000e-20,
                      2.520000e-20,
                      2.390000e-20,
                      2.000000e-20,
                      1.450000e-20,
                      1.150000e-20,
                      8.600000e-21,
                      6.400000e-21,
                      5.200000e-21,
                      3.600000e-21,
                      2.400000e-21,
                      1.800000e-21,
                      1.350000e-21
                ]

          ])

    effective[1, :] *= eff_coef
    excitation[1, :] *= exci_coef
    ionization[1, :] *= ion_coef

for v in vector_with_errors:
    print(v)

for p in X:
    print(p)

y = vector_with_errors

reg = LinearRegression().fit(X, y)

#out = reg.predict(np.array([[1.32, 1.3, 1.07]]))

#print(out)