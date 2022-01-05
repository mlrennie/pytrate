# pytrate
A python software package for analyzing titration data, with an emphasis 
on combined analysis of multi-titrations.  Based on and inspired by pytc from
the Harms Lab (see https://doi.org/10.1021/acs.biochem.7b01264; 
https://github.com/harmslab/pytc). Performs Bayesian fitting. 
Designed for fitting arbitrary models and arbitrary titration setups.

## Clone API from github

```
git clone https://github.com/mlrennie/pytrate
cd pytrate
python3 setup.py install
```

## Data file format

Input titration data should be csv format with 3 header lines. The 
first two define the units of the measurements, the third lists the 
data columns - concentrations followed by measured observables e.g.:
```
Conc units:,M,
Obs units:,nMS,
Conc 1,Conc 2,Conc 3,Obs 1,Obs 2,Obs 3,Obs 4,Obs 5
```

Following this are the numerical values for each of the datapoints, e.g.
```
0.,    5e-6,5e-6,0.559139784946237,0.,0.440860215053763,0.,0.
2.5e-6,5e-6,5e-6,0.449197860962567,0.,0.406417112299465,0.0160427807486631,0.128342245989305
5e-6,  5e-6,5e-6,0.39622641509434,0.00943396226415094,0.292452830188679,0.0283018867924528,0.273584905660377
10e-6, 5e-6,5e-6,0.130952380952381,0.0595238095238095,0.0595238095238095,0.0952380952380952,0.654761904761905
20e-6, 5e-6,5e-6,0.221052631578947,0.126315789473684,0.0105263157894737,0.0736842105263158,0.568421052631579
```

See also examples folder.
