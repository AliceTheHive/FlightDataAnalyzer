.. _DependencyTree:

===============
Dependency Tree
===============


Concept
-------

In brief: `To dynamically determine the processing order of dependencies to
satisfy the list of required parameters.`

Aircraft flight principles are generally the same across different aicraft
types. However the parameter set available differs considerably between
aircraft types and across different dataframes. This problem continues to
individual aircraft which may not have optional hardware installed or may
have a failed sensor resulting in a recorded parameter being invalid.

Often this can involve writing Aircraft Tail specific code exceptions when
deriving parameters to account for all these exceptional cases.

Keeping the Node modules small and ensuring a single node only serves a
specific purpose means that programmers can write re-usable code that is easy
to read.

The dependency tree takes away those issues by establishing which Nodes are
able to operate and in which order they need to be processed in order that
the hierarchical set of dependencies will be met.

The programmer need not worry about the order in which the code will be
executed. If a parameter is set as a dependency (optional or required) it
will have been evaluated by the time it enters the derive method.


What are a node's dependencies?
-------------------------------

The dependencies are defined as keyword arguments to the derive method which
exists on every Node ::

    from analysis_engine.node import DerivedParameterNode, P, KTI

    class NewParameter(DerivedParameterNode):
        def derive(self, first=P('First Dependency'), second=KTI('Second Dependency')):
            pass

Each dependency is specified as an instance of a node class with only the name argument provided. By adding the
dependency names into the **derive** method, the dependency tree can establish a link between the Node and the
parameters it can use for derivation. Additionally, the type of node classes specifies the node type of each dependency.

The following short-hand class names are used when defining dependency node types for brevity:

- P (DerivedParameterNode)
- M (MultistateDerivedParameterNode)
- S (SectionNode / FlightPhaseNode)
- KTI (KeyTimeInstanceNode)
- KPV (KeyPointValudeNode)
- A (Attribute)

Mach Example
~~~~~~~~~~~~

Many aircraft record `Mach`, but for those who do not we can dynamically
establish the Mach from the `Airspeed` and pressure altitude (`Altitude STD`).

    from analysis_engine.node import DerivedParameterNode, P

    class Mach(DerivedParameterNode):
        def derive(self, airspeed=P('Airspeed'), second=P('Altitude STD')):
            ...

If Mach is recorded within the data frame, Mach will already be available within the dependency tree at the start of processing and the Mach DerivedParameterNode will not be executed.

:py:class:`analysis_engine.derived_parameters.Mach`

For example, if one requests a Key Point Value of **Mach Max**, the dependency tree will establish whether or not it can provide all necessary dependencies.

    from analysis_engine.node import KeyPointValueNode, P

    class MachMax(KeyPointValueNode):
        def derive(self, mach=P('Mach')):
            ...

The dependency tree will establish that Mach is a requirement of Key Point Value **Mach Max** and if Mach is recorded within the data frame, no further calculations are
performed.

.. digraph:: MachRecorded

   "Mach Max (KPV)" -> "Mach (Recorded)"

If Mach is not recorded it will establish whether the Mach dependencies are
met (Airspeed and Altitude STD).

.. digraph:: MachDerived

   "Mach Max (KPV)" -> "Mach (Derived Parameter)" -> "Airspeed (Recorded)";
   "Mach (Derived Parameter)" -> "Altitude STD (Recorded)";


Can Operate
~~~~~~~~~~~

It may be possible to conduct the Node derivation (within the **derive**)
method without a full set of dependencies being available. The
**can_operate** class method allows the programmer to insert basic logic to
determine which of the available parameters (a subset of the dependencies
within the derive declaration) the Node can operate successfully with.::

    class NewParameter(Node):
        @classmethod
        def can_operate(cls, available):
            # New Parameter can work if the following two dependencies are available
            return 'Airspeed' in available and 'Altitude AAL' in available

        def derive(self, aspd=P('Airspeed'), gspd=P('Groundspeed'), alt=P('Altitude AAL')):
            # Check if Groundspeed is available
            if gspd:
                ...
            ...

So if Groundspeed is recorded in the LFL:

.. digraph:: NewParameterWithGroundspeed

   "New Parameter" -> "Airspeed";
   "New Parameter" -> "Groundspeed";
   "New Parameter" -> "Altitude AAL";

If Groundspeed is not recorded, the following will still work:

.. digraph:: NewParameterWithoutGroundspeed

   "New Parameter" -> "Airspeed";
   "New Parameter" -> "Altitude AAL";


See :ref:`can-operate` for more usage examples.


Debugging Can Operate
^^^^^^^^^^^^^^^^^^^^^

The :py:meth:`~analysis_engine.Node.NodeManager.operational` method of the
NodeManager calls the **can_operate** method on the classes when traversing
the dependency graph.

When a requested Node is not operating (does not execute the derive method)
you can quickly establish why without having to refer to the dependency tree
by inserting a breakpoint into the can_operate method. If there is no
can_operate method, temporarily add one with a pass statement to breakpoint
upon.::

    class NewParameter(Node):
        @classmethod
        def can_operate(cls, available):
            pass  # add a breakpoint here to inspect "available"

        def derive(self, ...):
            ...

..
    As an example, one may calculate a smoothed latitude and longitude location
    of the aircraft from the recorded Latitude and Longitude which may not have a
    very high resolution (causing a steppy track). Latitude Smoothed will depend
    on Latitude:

        Latitude Smoothed
        requires: Latitude

    In order to better increase the accuracy of the aircraft, some information about the Takeoff and Landing runway will help to pin-point the track onto the runway:

        Latitude Smoothed
        requires: Latitude
        optional: Takeoff Runway, Landing Runway

    The derived parameter will make the most out of the parameters provided - so
    if the Takeoff Runway isn't known, it will be smoothed without pinpointing
    the track to the runway.

    Some aircraft don't record their location, so instead we can use Heading and Airspeed to derive a track and then pinpoint this onto the runways:

        Latitude Smoothed
        requires: Latitude or (Heading and Airspeed and Latitude At Takeoff and Latitude At Landing)
        optional:



Graph Theory
------------

All Nodes (Derived Parameter Nodes, Attribute Nodes) are all objects which
can have dependencies upon other Nodes or LFL Parameters.

.. digraph:: MachMax

   "Mach Max" -> "Mach" -> "Airspeed";
   "Mach" -> "Altitude STD";


Each of these objects is a Node within a directional graph (`DiGraph`). The
edges of the graph represents the dependency of one Node upon another.


Processing Order
~~~~~~~~~~~~~~~~

The processing order is established by recursively traversing down the
DiGraph using Depth First Search.

:py:func:`analysis_engine.dependency_graph.dependencies3`

As each Node is encountered, if it has dependencies we recurse into each
dependency to determine whether the level below is operational. If deemed
operational, the Node is added to the set of active_nodes (so that we do not
process the node again) and appended to the processing order.

The **root** node is a special node which defines the starting point of the
DiGraph for traversal of the dependency tree. It points to the top level
parameters (those which have no predecessors).

To evaluate a Key Point Value **Mach Max** and another "Mach At Flap
Extension" the following graph may be created:

.. digraph::

   "root" -> "Mach Max" -> "Mach" -> "Airspeed";
   "Mach" -> "Altitude STD";
   "root" -> "Mach At Flap Extension" -> "Mach";
   "Mach At Flap Extension" -> "Flap";


This is the processing order:

.. digraph:: MachMaxProcessingOrder

   "7: root" -> "4: Mach Max" -> "3: Mach" -> "1: Airspeed";
   "3: Mach" -> "2: Altitude STD";
   "7: root" -> "6: Mach At Flap Extension" -> "3: Mach";
   "6: Mach At Flap Extension" -> "5: Flap";


Spanning Tree
~~~~~~~~~~~~~

The Spanning Tree is a copy of the original Graph, excluding the inactive
Nodes. It represents the actual tree to be used for analysis. These may be
inactive due to being inoperable (the dependencies do not satisfy the
can_operate method) or not being available (the NodeManager does not contain
them, normally due to not being recorded in the LFL but possibly due to a
naming error).


Visualising the Tree
~~~~~~~~~~~~~~~~~~~~

The graph can be visualised in a few ways.

**print_tree()** - A simple textual. output printed to screen which uses
`indent_tree` to structure the node hierarchy in a tree-like view. The
starting node can be selected, providing the ability to print subsets of the
tree.

:py:func:`~analysis_engine.dependency_graph.print_tree`

**draw_graph()** - This requires `pygraphviz` and therefore `Graphviz` to be
installed.

:py:func:`analysis_engine.dependency_graph.draw_graph`


The `FlightDataParameterTree` tool can also be used to visualise the
dependency tree. This can be easier to understand when your tree is quite
large (often the case!).

The numeric before the Node name represents the Nodes position in the
processing order.

Colours are used to represent the different types of parameters.

.. note::

    Networkx was chosen over pygraph due to its more pythonic implementation.


.. warning::

    Circular dependencies are complicated to comprehend, so try to avoid creating them!
    Example:


Circular Dependencies
~~~~~~~~~~~~~~~~~~~~~

.. digraph:: circular

    "Heading True" -> "Magnetic Variation"
    "Heading True" -> "Heading"
    "Heading" -> "Heading True"
    "Heading" -> "Magnetic Variation"

DiGraphs support edges from A -> B and B -> A, this would normally cause
infinite recursion when resolving the processing order using depth first
searches. The dependency tree resolves this by keeping track of nodes it has
seen already in the depth first traversal, when encountering a node already
seen it will declare a warning of a circular dependency and declare the node
as inoperable.


.. How to view / identify problems
