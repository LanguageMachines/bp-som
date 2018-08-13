# BP-SOM

BP-SOM implements a combination of a multilayered feedforward network (MFN) and one or more self-organising maps (SOMs);
each hidden layer of the MFN has its corresponding SOM. Training a BP-SOM is a combination of supervised learning with the
traditional back-propagation (BP) learning rule guided by clustering information in the SOMs.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

<!---
### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```
--->

### Installing

You can compile the code with

```
sh bootstrap.sh
./configure [--prefix=place-to-install]
make [install]
```
<!---
And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running an experiment


--->


### A test run

bpsom takes a configuration file as argument. An example config file, example-grapho.prm, can be found in the examples directory.
You can run bpsom on the example by going to the examples directory and invoking bpsom:

```
cd examples
bpsom example-grapho.prm
```

bpsom will start producing output such as:

```
1/100 LRN=1 MAT=1 MSE=0.935 CE= 39.23 USE SOM1=  0.0 
1/100 LRN=0 MAT=1 MSE=0.813 CE= 26.41 USE SOM1=  0.0 
1/100 LRN=0 MAT=2 MSE=0.813 CE= 26.48 USE SOM1=  0.0 *
1/100 LRN=0 MAT=3 MSE=0.813 CE= 26.43 USE SOM1=  0.0 
```

1/100 means it is in the first of a maximum of 100 epochs. During
learning (LRN=1) it monitors its progress on the training data
(MAT=1), measuring a mean squared error of 0.935 and a classification
error of 39.23% (MSE=0.935 CE= 39.23). The SOM, aligned with the single
hidden layer, has not been adding its error signals into the
back-propagating error yet (USE SOM1= 0.0).

The next three lines provide scores on the three types of data
provided, while activation is only fed forward (i.e., training is off,
LRN=0): the scores on training data (MAT=1), development data (MAT=2),
and test data (MAT=3). Development data is used to determine whether
the early stopping criterion applies (a predetermined number of epochs
is reached in which the CE on the development data does not
improve). At the first epoch and at any future epoch in which the CE
improves, the network is saved (indicated by the asterisk, *).

When the stopping criterion is reached, or when 100 epochs have
finished, the following output is generated:

```
opening exp-grapho-bpsom-a10.bst
66/100 LRN=0 MAT=3 MSE=0.578 CE=  9.89 USE SOM1=  0.0 
16/100 LRN=0 MAT=1 MSE=0.656 CE=  7.21 USE SOM1=  0.0 
16/100 LRN=0 MAT=3 MSE=0.661 CE=  7.80 USE SOM1=  0.0 
```

This means that the lowest MSE was attained after epoch 16, when
classification error on the test data was 7.80%. At that point, the
SOM was not used. The network's state at epoch 16 is saved in the file
exp-grapho-bpsom-a10.bst .

If a hidden unit becomes inactive below a threshold, it gets
pruned. In the output this is determined on the development set, and
indicated as follows:

```
20/100 LRN=1 MAT=1 MSE=0.619 CE=  9.81 USE SOM1= 42.3 
20/100 LRN=0 MAT=1 MSE=0.612 CE=  9.26 USE SOM1=  0.0 
20/100 LRN=0 MAT=2 MSE=0.611 CE=  9.25 USE SOM1=  0.0 * 1:-7 1:-8
20/100 LRN=0 MAT=3 MSE=0.614 CE=  9.65 USE SOM1=  0.0 
```

This means that at epoch 20, the seventh and eighth hidden units of
layer one were pruned.

### Inspecting the log file

During a full experimental run involving learning and testing, a
detailed log file is produced which can be inspected afterwards. This
allows ascertaining whether and to what extent the SOM became
organized during learning, and how it influenced hidden unit
activation behavior. After a dump of the full configuration and the
distribution of classes in the training data, the log file contains
for each epoch

* the learning progress as also displayed in the console as stdout;
* the majority class labeling for the SOM(s);
* the average hidden unit activations and their standard deviations
for all hidden layers.

The majority class labeling displays for each SOM unit the class that
is most prominently mapped onto that unit:

```
Class labeling SOM: 1

      1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
     53  73  37   0  64  53  46  71  74  98  99  78  47 100  93  90  82  78  91  82
  1 -{- -{- -@-     --- -k- -k- -s- -s- -s- -s- -s- -z- -f- -f- -f- -n- -n- -n- -n-
    964 1577 1484   0 1737 794 1229 1766 166 3643 2623 561 2366  26 3203 1944 764 1899 1326 1853

...

     99  55  83   0  83  95  96  56  94  62  98  98 100  49  56  96  56  84   0  45
 20 --- -s- -z-     -d- -d- -t- --- -I- -I- -I- -I- -I- -@- --- --- --- ---     -#-
    2045  70 6281   0 5789 2020 6645 398 5665 303 2931 4736  15 1653 3251 5861  46 2550   0 1989
```

If we inspect the second SOM unit in the last line (unit 20,2), we see that 70
training vectors map onto this SOM unit, 55 of which are labeled with
class "-s-". We also see that the fourth unit in the same line (unit
20,4) is empty; no training vectors are mapped on it.

<!---
### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).

--->

## Authors

* **Ton Weijters** - *Initial work in 1990s*
* **Antal van den Bosch** - *additional work in 1990s, revival in 2018* -  [antalvdb](https://github.com/antalvdb)

<!---
See also the list of [contributors](https://github.com/your/project/contributors) who participate in this project.
--->

## Further reading

Further reading:

* Weijters, A. (1995). The BP-SOM architecture and learning algorithm.
  *Neural Processing Letters*, **2:6**, pp. 13-16.

* Weijters, A., Van den Bosch, A., and Van den Herik,
  H.J. (1997). Behavioural aspects of combining back-propagation and
  self-organizing maps. *Connection
    Science*, **9:3**, pp. 253-252.

* Weijters, A., Van den Herik, H.J., Van den Bosch, A., and
Postma, E. (1997). [Avoiding overfitting with BP-SOM.](https://www.ijcai.org/Proceedings/97-2/Papers/051.pdf) In
*Proceedings of the Fifteenth International Joint Conference on
    Artificial Intellgence*, IJCAI-97, pp. 1140-1145.

* Weijters, A., Van den Bosch, A., and Van den
  Herik, H.J. (1998). [Interpretable neural networks with BP-SOM.](https://link.springer.com/content/pdf/10.1007%2FBFb0026711.pdf) In
  C. Nedellec and C. Rouveirol (Eds.), *Machine Learning:
    ECML-98*, Lecture Notes in Artificial Intelligence,
  Vol. 1398. Berlin: Springer, pp. 406-411.

* Weijters, A., and Van den Bosch, A. (1998). Interpretable
  Neural Networks with BP-SOM. In A.P. del Pobil, J. Mira and M. Ali
  (Eds.), *Tasks and Methods in Applied Artificial Intelligence:
    Proceedings of the 11th International Conference on Industrial and
    Engineering Applications of Artificial Intelligence and Expert
    Systems*, Vol II, Lecture Notes in Artificial Intelligence, Vol.
  1416. Berlin: Springer, pp 564-573.

* Weijters, A., and Van den Bosch, A. (1999). [Interpreting
knowledge representations in BP-SOM.](https://www.jstage.jst.go.jp/article/bhmk1974/26/1/26_1_107/_pdf)
*Behaviormetrika*, **26:1**, pp. 107-128.

* Weijters, A., Van den Bosch, A., and Postma, E. (2000). [Learning
  statistically neutral tasks without expert guidance.](https://papers.nips.cc/paper/1780-learning-statistically-neutral-tasks-without-expert-guidance.pdf) In: S.A. Solla,
  T.K. Leen and K.R. Muller.), *Advances in Neural
    Information Processing*, Vol. 12. Cambridge, MA: The MIT Press,
  pp. 73-79.


## License

This project is licensed under the GNU Public License version 3.0 - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Ton Weijters for coining and developing the BP-SOM idea and for graciously allowing the code to be revived;
* Eric Postma and Jaap van den Herik for support and discussions, and co-writing some of the original papers;
* The Language Machine team (the Lamas) for the revival of the project 20 years later.
