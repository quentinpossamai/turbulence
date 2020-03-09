from ai2thor.controller import Controller
from PIL import Image

RANDOM_SEED = 0

# Initialisation of a controller and in the same time a scene
c = Controller(scene='FloorPlan28', gridSize=0.25, renderDepthImage=False, renderClassImage=False,
               renderObjectImage=False, width=640, height=480, agentMode='bot')

# Initialisation of a scene only
c.reset(scene='FloorPlan28')

# c.step('Initialize', gridSize=0.25, agentCount=1)

# Place randomly pickupable objects
c.step(action='InitialRandomSpawn', randomSeed=RANDOM_SEED, numPlacementAttemps=5)

# Randomize the state of all the toggleable objects
c.step(action='RandomToggleStateOfAllObjects', randomSeed=RANDOM_SEED)

# Randomize the state of all specific states that can have objects
c.step(action='RandomToggleSpecificState', randomSeed=RANDOM_SEED,
       StateChange=['CanOpen', 'CanToggleOnOff', 'CanBeSliced', 'CanBeCooked', 'CanBreak', 'CanBeDirty',
                    'CanBeUsedUp'])

event = c.step(action='MoveAhead')
pos_agent = event.metadata['agent']['position']

event2 = c.step(dict(action='AddThirdPartyCamera', rotation=dict(x=0, y=0, z=90), position=pos_agent))

event = c.step(action='MoveAhead')
sum(event2.third_party_camera_frames[0].flatten())

# event = controller.step(action='GetReachablePositions')
# e = event.metadata['actionReturn']

a = event.frame
b = event.depth_frame
c1 = event.class_segmentation_frame
d = event.instance_segmentation_frame

img1 = Image.fromarray(a, 'RGB')
img1.show()
img2 = Image.fromarray(b, 'RGB')
img2.show()
img3 = Image.fromarray(c1, 'RGB')
img3.show()
img4 = Image.fromarray(d, 'RGB')
img4.show()

c.stop()
