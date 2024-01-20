## Project AUTOMATIC
Project "AUTOMATIC: Analysis of the relationship between the AUTOnoMic nervous system and cerebral AutoregulaTion using maChine learning approach" is financed by grant SONATA-18 National Science Center (UMO-2022/47/D/ST7/00229)


This project aims to fill a gap in the methodology of analysing the temporal correlation between cerebral autoregulation and ANS taking into account the dynamic of this association. The main goals of the project can be summarised as follows:

Aim 1. Clear-cut experimental evidence of the character of the association between autonomic response variables and cerebral autoregulation in healthy volunteers.

Aim 2. The characteristic of the contemporaneous relationship between autonomic response variables and cerebral autoregulation using an adaptation of advanced time-series data analysis methods.

Aim 3. A robust understanding of how cerebral autoregulation and ANS are interconnected in patients with intracranial pathologies


## Aim of the repository
The aim of this repository is to provide the code for phase rectified signal averaging (PRSA) and to applay it to the data from the project AUTOMATIC. The code is divided into two parts:
1. prep.py - preprocessing of the data, and displaying its properties
2. prsa.py - all functions connected with PRSA

## Data
The data used in this project is not public. It comes from healthy volunteers aged 20-25 years old.

## Alghoritm
The alghoritm of PRSA is based on the following steps:
1. Finding peaks in the signal

![signal_sample](images/signal_sample.png)

2. Determing anhor points on rr-intervals plot

![rr_plot](images/rr_plot.png)

3. Taking windows around anchor points and averaging them

![anchors_windows](images/anchors_windows.png)

4. Calcultaing AC and DC component of the corresponding averaged signal using following formula: AC and DC = [RR(0) + RR(1) - RR(-1) - RR(-2)]/4

![averaged_signal](images/averaged_signal.png)