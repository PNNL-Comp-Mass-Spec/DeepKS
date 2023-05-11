"""Default entry point for running examples."""

from . import examples
from ..tools.splash.write_splash import write_splash

write_splash("examples.spash")

examples._main()
