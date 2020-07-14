########################################################################
#
#   Self Adapting User Profiles
#   Tim Vande Walle
#   2019-2020
#   Thesis VUB
#   promotor: Olga De Troyer
#
#   This is the main file and contains functionality to start the algorithms.
#   The algorithms are in other files.
#
########################################################################


########################################################################
#   Imports
########################################################################
import random

import configuration
import util

import analyse

import contentbased
import linreg
import polreg
import randfor
import logreg
import logregpca
import logregpcafeatsel
import nb


########################################################################
#   Functions - app Logic
########################################################################
def menu():
    choice = True
    while choice:
        print("""
        0:  Preprocess and analyse data
        1:  N/A Run Genetic Algorithm
        2:  Run Content-based Algorithm
        3:  Run Predictive Model Linear Regression              (automatic feature selection)
        4:  Run Predictive Model Polynomial Regression          (automatic feature selection)
        5:  Run Predictive Model Random Forest Parameter Tuning
        6:  Run Predictive Model Random Forest 
        7:  Run Predictive Model Logistic Regression            (without PCA)
        8:  Run Predictive Model Logistic Regression            (with PCA)
        9:  Run Predictive Model Logistic Regression            (with PCA + automatic feature selection)
        10: Run Predictive Model Naive Bayes             
        
        99: Run All
        999: Exit
        """)

        choice = input("Which algorithm would you like to run? ")

        if choice == "0":
            run_preprocess()
            exitProgram()

        if choice == "1":
            run_geneticAlgorithm()
            exitProgram()
        elif choice == "2":
            run_contentBasedAlgorithm()
            exitProgram()
            
        elif choice == "3":
            run_predictiveModelLinReg()
            exitProgram()
        elif choice == "4":
            run_predictiveModelPolReg()
            exitProgram()
        elif choice == "5":
            run_predictiveModelRandForParTuning()
            exitProgram()
        elif choice == "6":
            run_predictiveModelRandFor()
            exitProgram()
        elif choice == "7":
            run_predictiveModelLogReg()
            exitProgram()
        elif choice == "8":
            run_predictiveModelLogRegPCA()
            exitProgram()
        elif choice == "9":
            run_predictiveModelLogRegPCAFeatSel()
            exitProgram()
        elif choice == "10":
            run_predictiveModelNB()
            exitProgram()

        elif choice == "99":
            run_all()
            exitProgram()
        elif choice == "999":
            exitProgram()
        elif choice != "":
            print("Invalid option chosen. PLease try again.")

def run_preprocess():
    print()
    print("########################################################################")
    print("running preprocessing and analysing data")
    analyse.run()

def run_geneticAlgorithm():
    print()
    print("########################################################################")
    print("running genetic algorithm")
    print("NOT YET IMPLEMENTED")

def run_contentBasedAlgorithm():
    print()
    print("########################################################################")
    print("running content-based algorithm")
    contentbased.run()

def run_predictiveModelLinReg():
    print()
    print("########################################################################")
    print("running predictive model - linear regression")
    linreg.run()

def run_predictiveModelPolReg():
    print()
    print("########################################################################")
    print("running predictive model - polynomial regression")
    polreg.run()

def run_predictiveModelRandFor():
    print()
    print("########################################################################")
    print("running predictive model - random forest")
    randfor.run()

def run_predictiveModelRandForParTuning():
    print()
    print("########################################################################")
    print("running predictive model - random forest parameter searching & tuning")

def run_predictiveModelLogReg():
    print()
    print("########################################################################")
    print("running predictive model - logistic regression")
    logreg.run()

def run_predictiveModelLogRegPCA():
    print()
    print("########################################################################")
    print("running predictive model - logistic regression - with PCA")
    logregpca.run()

def run_predictiveModelLogRegPCAFeatSel():
    print()
    print("########################################################################")
    print("running predictive model - logistic regression - with PCA + feature selection")
    logregpcafeatsel.run()

def run_predictiveModelNB():
    print()
    print("########################################################################")
    print("running predictive model - naive bayes")
    nb.run()


def run_all():
    print("running all algorithms")
    print("not yet configured")

def exitProgram():
    print()
    print("########################################################################")
    print("exiting program")
    exit()

########################################################################
#   MAIN
########################################################################
print("Self Adapting User Profiles")
menu()
