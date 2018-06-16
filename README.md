# Project Title

BP-SOM implements a combination of a multilayered feedforward network (MFN) and one or more self-organising maps (SOMs);
each hidden layer of the MFN has its corresponding SOM. Training a BP-SOM is a combination of supervised learning with the
traditional back-propagation (BP) learning rule guided by clustering information in the SOMs.

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

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

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

## License

This project is licensed under the GNU Public License version 3.0 - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Ton Weijters for coining and developing the BP-SOM idea and for graciously allowing the code to be revived;
* Eric Postma and Jaap van den Herik for support and discussions, and co-writing some of the original papers;
* The Language Machine team (the Lamas) for the revival of the project 20 years later.
