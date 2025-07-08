Hello! I'm not great at formatting markdown, but here is my attempt at a ReadME!

So a few things to note: 
- This code (the main one, NOT the client one) should be able to interact with another instance of itself through socket connection. This is how users are meant to interact.
- The yellow detector can get messed up in poor lighting so make sure the mask shows that the yellow object is the only yellow observable detected.
- There are some modules that need to be installed, just refer to the import list in the python files to check.
- Lastly, if you want to run the simulation on one PC, you can't use 2 instances of the main code, so you need to boot up one client and one main code. This is because multiple programs cannot access the same camera.
- Feel free to tweak this or contribute!
