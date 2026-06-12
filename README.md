# GAME PHYSICS ENGINE

## About
This project is for the final project of Harvard University's CS50x Introduction to Computer Science for the year 2026.

It's a simple physics engine with one working demo. Albeit, it has the components to produce other demos. The demo provided is showcases some basic features of the engine - forces, contacts and shapes.

The engine itself is heavily taken from Ian Millington's book *Game Physics Engine Development*, but does incorporate my own interpretation of some ideas.
> Watch the demo [here](https://youtu.be/Hh4ZnpWwukk)

---

## Features and implementation

### Data structures and Algorithms
- Separating-axis theorem test
- Sutherland-Hodgman Algorithm
- Bounding Sphere Hierarchy
- Binary Spatial Partitions (although not used in the demo)

### Physics and mathematics
- Euler's method
- Sequential Impulses (based on Box2D)
- Gyroscopic effect is kept in

#### note
Some other demos exist in previous versions of the repository, but have been removed due to consideration of the **Academic Honesty Policy**.

--- 

## Usage
From the root directory, you may run:
> python demos/spring_and_buoyancy.py

It's right mouse button to control the camera angle, and middle mouse button to move the camera.

---

## References
- [Erin Catto 2006](https://box2d.org/files/ErinCatto_SequentialImpulses_GDC2006.pdf)
- [Erin Catto 2014](https://box2d.org/files/ErinCatto_UnderstandingConstraints_GDC2014.pdf)
- [Dirk Gregorious - Robust Contact Creation](https://media.steampowered.com/apps/valve/2015/DirkGregorius_Contacts.pdf)
- Ian Millington, Game Physics Engine Development: 2nd Edition
- [This video explains inertia tensors well](https://www.youtube.com/watch?v=GYc99lMdcFE)
- Google AI Studio and GitHub Copilot - "*I've commented where there is direct involvement, but large majority is assisted learning and debugging*"

---

## SRC/physics - More details
### Quaternions, integrator and transform
#### Quaternions
This file contains the standard operations needed for the engine. It uses the scipy.spatial.rotation library to simplicity. It also has the added benefit of normalizing the quaternions, which prevent drift over time. add_scaled_vector is a method called when adjusting the orientation of rigid bodies, used in integrator.py. I chose quaternions to avoid gimbal lock associated with Euler angles. However, ursina mainly uses euler angles so this came with a drawback.

#### Integrator
This file contains one class and one method for integration. Why a standalone file? - To allow modularity. I considered using RK4 or Verlet, however, ultimately landed on the Euler Method (or more so I never changed). It's relatively light and fast compared to RK4, and I didn't feel comfortable using methods where I hadn't thoroughly understood the underlying mathematics. Due to the computational cost and black-box feel, I decided to stick to Euler's method, unless I plan on revisiting the dzhanibekov effect, Euler's method provides satisfying results.

#### Transform
This file contains the standard conversions ('transformations') needed to go between local and world space. Why? Any transformation in 3D can be simplified to a translation and rotation (or at least according to this Chasles guy). Thus, the transformation can be simplified to a 4x4 homogeneous transformation matrix. The class has a classmethod which calls itself to build a new matrix, which is important for rigidbody.py

---

### Force generators and Force registry
#### Force generators
This file contains a collection of 'force generators'. From my prior implementation of force generators on particles, I learnt about abstract classes and factory patterns. By using ABC classes, it effectively ensures each generators follow a strict 'contract' and behave roughly the same. I debated for a while between using Protocols or ABC - I liked the flexibility Protocol provides. However, I decided that the nature of force generators should be enforced strictly and that any gains from Protocol wouldn't be felt in this instance. Each subclass of 'ForceGenerator' must have an abstractmethod of update_force, allowing my 'world' class to iterate through the registry cleaner.

#### Force registry
This file contains the 'force registry'. The key component of the 'ForceRegistry' class is the registrations list. Most methods of this class manage this list - deregister/ register. The exception being the 'update_force' method. This allows the class to effectively act as a manager on forces in the engine and sets up a platform so that the 'world' never has to 'see' the backend. In world, as we will see later, the run method only has to call the update_force method, which then updates every rigid body and force generator pair such that appropriate forces are applied to each rigid body. It was around this time I learnt more about classes and decorators. I chose to use dataclasses to allow for groupings of relevant data ('ForceRegistrant' and '_RegistrationKey'), giving the benefit of better legibility and reducing parameters. Because of ABC used in force_generator.py, the class never actually knows what it's dealing with, while everything stays consistent. Additionally, ids were used to keep track of each pair. I chose to use NamedTuple because of its immutability, which ties nicely to the nature of its task.

---

### Rigid Body and World
#### Rigid Body
This file contains the class RigidBody. The class stores its state (velocity, omega, position, ...) but also includes methods to affect its state. This relates back to the force generators. Essentially, the pipe goes like this: select force generators -> world -> registry stores, then when update_force is called -> registry goes through each force generator and updates force -> force generators run unique calculations -> uses the methods in RigidBody to alter its state (-> integrate). I chose to cache the transform matrix, inertia tensors and inverse inertia tensors to potentially save on performance using a dirty flag. Additionally, a sleep system is used to further save on performance, which will be important later on.

#### World
This file contains the world class and is the primary orchestrator between each component. 'run_physics' is sequence dependent and must be executed with a certain order. Initially, there was a design decision I had to make between (1) the world owns each object (2) the object owns itself, I think. I went with (1) because it is cleaner and provides better modularity with each component of the system. The aforementioned 'force chain' lives within run_physics, and several other methods which allows the registration/ deregistration of rigid bodies (the world holds the objects). Additionally, it has methods which allows the registrations of other primitives and shapes, as well as methods for contact detection and resolution, which we will get into later.

---

### Broad phase contact detection
#### Broad Phase
This file contains the Protocol classes and dataclasses involved in broad phase detection

#### BSH - Bounding Sphere Heirarchy
This file contains the classes and methods needed for BSH. Why BSH? It's easier than BVH (specifically AABB or any of the sorts) because spheres only have one check for collision. BSH follows a binary tree structure, thus there is a class for the node, sphere and tree. While it may seem fancy (at least to me at the time), the process is actually quite simple. The system still checks for collision between every existing pair, but the tree cuts down the number of choices. The general process can be summarized as such: Build tree -> Find collisions -> return **candidate pairs** (I cannot further stress that you **have** to build the tree first). 

Whole pipeline to get a broader understanding:
For each body, *world* adds shapes to a body, which I will explain later. For each shape on a body, a bounding sphere is created respectively. Afterwards, each body is inserted into the BSHTree. Also note, that for some reason volume can mean literally volume or just the spheres

##### Insertion
The goal is to minimize the **growth** of r (because its a sphere). This minimization is done such that each object is grouped closer to one another. For example, if we grouped ourselves with an object across the room compared to an object closer to us, then there would be tons of empty space between that is included in the group. If such is the case, then the system would register anything between as a collision, even if no collision occurs. Because every child node is a leaf, by inserting a leaf body, it means somewhere in the tree, a child becomes a parent and the new insertion is the child. Thus, not only is a new bounding sphere generated to group the new leaf and the old one, but the nodes are also shuffled around to ensure that only children nodes are leaves. Finally, each sphere is refitted **upwards**

##### Removal
Essentially, there should be **no** 'thin' branches in the BSHTree. When you remove a leaf, the parent node will only have one child, which is not desired. So the parent is demoted and becomes the removed leaf's sibling (or conversely, the removed leaf's sibling gets promoted to be the parent's sibling). Afterwards you refit each sphere **upwards**.

##### overlapping
For the actual detection bit, it's a recursive algorithm. The key property is that if two parents don't overlap, then neither do their children. Since bodies only exist in leaf nodes, the return condition is if the children nodes contain not nothing (or bodies). The algorithm chooses the side which is not a leaf and has the greatest radius (sphere property) then compares its children to its sibling recursively. It's not about if they overlap, it's about if they don't because that means one entire branch gone.

#### BSP - Binary Space Partitions
This file contains components of a BSP - BSH hybrid approach. Similar to BSH, the BSPTree is built first. I chose to use an iterative method (TODO stack) this time because I heard it's better practice. It's another binary tree, where each plane will have a front side and a back side, so two children nodes. For insertion and removal, I used an iterative approach. Where BSP differs from BSH is in collision detection. Instead of looking for overlaps, all bodies in any leaves are candidates. Thus, there is a method to collect all leaves, which I used in hybrid with BSH to refine detection. One problem was duplication. If you asked 'what happens to objects that are in between the front and back of a plane?', then that object is actually put into both the front and back. By using a set and the id of each body. I check ignore any pairs already seen. 'What about order? says the mathematician' {x,y} : x < y.

---

### Narrow Phase
#### narrow_phase.py
This file contains hand set methods and dataclasses for narrow phase collision detection between basic shapes. This system uses a double dispatch approach to handle collisions between different combination of objects - sphere-box, sphere-plane, sphere-sphere. The issue is that single dispatch solely depends on the calling object, thus, just increase it!! Using a double dispatch system makes *world* a lot cleaner, *world* never knows what type of contact it's actually dealing with - encapsulation!

To avoid writing too much, here is a short table:


|        | sphere                         | box                        | plane                                           |
|--------|--------------------------------|----------------------------|-------------------------------------------------|
| sphere | compare radius sum to distance | treat it as vertex to face | check dot product, set contact point to surface |
| box    | -                              | SAT + Sutherland-Hodgman   | modified SAT                                    |
| Plane  | -                              | -                          | no result                                       |

Let me highlight some notable areas.
- lines 228-236, when magnitude is near 0, the sphere's center is essentially at the box's center. Since penetration is defined as the shortest distance out, the system finds the face of such and resolves it appropriately, handling the degenerate case
- lines 462-472, from the center, going along the half sizes will bring you to any vertex, which implies that walking in the direction towards the plane from the center will bring you to the vertex closest to the plane. Thus, giving a test for an early exit.
- SAT + Sutherland-Hodgman. ClipVertex established in contact.py allows the vertices to be tracked as well as positions, allowing for the construction of the manifold. I can't prove the SAT, but essentially the axis of the minimum penetration is the penetration axis, giving penetration and contact normal. I've just put all 15 axis that need to be taken into consideration into one array: first 6 are face normals, remaining are edges. Since faces are more stable, I've biased contacts towards face normals while limiting degenerate parallel cases. You might ask why I used Sutherland-Hodgman for box-box (face-vertex) instead of the approach for box-plane (face-vertex as well), if so, then I'll explain later. Anyways, the algorithm 'trims' the incident box such that a manifold is obtained. To do such, the system imagines four infinitely tall walls projected from the reference face edges parallel to the contact normal, then everything outside those walls are ignored. The 'entry' and 'exit' points of the lines connecting to points outside the frame become contact points. It's quite ingenious of me to use the basis vectors as normals to those planes. Another question to ask, is what is going on between lines 601-654. With a box, it's possible to get up to 8 contact points from the clipping process. However, a box probably only needs 4 points to be stable (think of tables and chairs), the algorithm chooses 3 points that maximize surface area, with 1 of the points being the worst penetration.

---

### Contact and Contact solver
#### contact.py
This file contains the dataclasses for contact_solver.py and narrow_phase.py, such as ContactManifold and ClipVertex. The heirarchy is as such: ContactData > ContactManifold > Contact, where ContactManifold contains Contact and so on. There are also helper methods that is useful for box-box collisions and *world*

#### contact_solver.py
This file contains the contact resolution pipeline and all methods and algorithms associated. The current solver goes through three steps: Prepare data -> resolve velocities -> resolve penetrations. I cache some data because iterative calculation of them is too computationally intensive, and the small time step means that the engine is approximately believable, even with the constant values. If you are interested in performance, then there is a section about it at the end. The current approach uses a sequential impulse solver. Why? I wanted to understand more about the maths behind XPBD or GJK+EPA before implementing those algorithms. 

---

### Materials
#### material.py
This file contains basic material types for friction and restitution values.

---
