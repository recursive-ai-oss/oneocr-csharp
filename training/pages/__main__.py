"""`python -m pages` entry point. The worker code lives in pages.build so that spawned processes can import it."""

from .build import main

main()
