# NEAT Template Python3
NEAT stands for "Evolving Neural Networks through Augmenting Topologies"

Original version in JavaScript programmed by Code-Bullet at [here](https://github.com/Code-Bullet/NEAT-Template-JavaScript).

Ported by CoolCat467 to Python3.11. Based of Code-Bullet's Javascript.


# Version Information:
Version 0.0.0: Inititial Publc Release

Version 1.0.0: Implemented Save-Load feature [as requested in the original project](https://github.com/Code-Bullet/NEAT-Template-JavaScript/issues/1)

Version 1.1.0: Added doccumentation, added BaseEntity object that BasePlayer now inherits, changed some other small things. Removed un-needed dependancy on threading module, as it was not used.

Version 2.0.1: Various changes to make pylint happy and minor fixes, removed pygame requirements.

Version 2.1.0: Fix type annotations, make code cleaner, rename `ConnHist` to `History`, and add `World` class. With the new type annotation fixes, it should be possible to compile this module to a C extension using [mypyc](https://mypyc.readthedocs.io/en/latest/introduction.html) for even faster code.


P.S.: If Code-Bullet ever sees this, your youtube videos are great.
