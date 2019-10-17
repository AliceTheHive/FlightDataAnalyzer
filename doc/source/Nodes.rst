.. _Nodes:

=====
Nodes
=====

Each derived parameter and key point value etc. is a Node. Each Node forms a
part of the :ref:`DependencyTree`.

There are certain attributes which all nodes have in common.

* frequency - `in Hz`
* offset - `in secs`
* derive method - `where the derivation work is performed`
* get_name  - `to auto-generate the "Nice Name" from "class NiceName"`
* get_aligned - `returns a version of itself aligned to another node`


Node Types
~~~~~~~~~~

.. py:module:: analysis_engine.node

.. inheritance-diagram:: KeyTimeInstanceNode KeyPointValueNode FlightPhaseNode DerivedParameterNode FlightAttributeNode ApproachNode MultistateDerivedParameterNode
   :parts: 1

The following Nodes are documented here:

* :ref:`ListNode` - provides list like functionality for storing multiple occurrences of a KeyPointValue or KeyTimeInstance within a Node
* :ref:`FormattedNameNode` - provides the ability for each occurrence to have a name formatted from variables defined by class variables in advance
* :ref:`KeyTimeInstanceNode` - important time instances within the flight
* :ref:`KeyPointValueNode` - individual value measurements taken from the flight
* :ref:`FlightPhaseNode` - defined intervals of the flight
* :ref:`DerivedParameterNode` - arrays derived from other parameters
* :ref:`FlightAttributeNode` - pythonic objects derived from the flight such as takeoff airport
* :ref:`ApproachNode` - a list of approaches which can account for multiple approaches into different airports


Node Names
~~~~~~~~~~

By default the name of a node, returned by `node.get_name()`, is automatically generated from its class name by adding spaces between camel-cased words, for instance:

    from analysis_engine.node import Node

    class NewParameter(Node):
        ...

In the above example, `NewParameter.get_name()` would return `'New Parameter'`. Automatic naming is suitable for the majority of node names, though the simple name generation method may be insufficient for more complex names.

    class AltitudeAAL(DerivedParameterNode):
        ...

In the above example, `AltitudeAAL.get_name()` would return `Altitude Aal` and lose the capitalisation of the AAL achronym. Another problem with automatic name generation is where the name should include characters which are not valid Python syntax for a class name, e.g. `Brake (*) Temp Max`. In these cases it is possible to manually set a node's name by defining the static class attribute `name`:

    class Brake_TempMax(DerivedParameterNode):
        name = 'Brake (*) Temp Max'
        ...

First Available Dependency
~~~~~~~~~~~~~~~~~~~~~~~~~~

By default we align all parameters to the first available dependency.::

    from analysis_engine.node import P, Node

    class NewParameter(Node):
        ##align = True  # default
        def derive(self, a=P('A')):
            pass

A fresh instance of NewParameter has the default Node frequency (1.0 Hz) and offset (0 secs)::

    >>> new = NewParameter()
    >>> new
    NewParameter('New Parameter', 1.0, 0)

The **get_derived** method takes the list of dependencies and prepares them
for use (aligning them as required) for the Node's **derive** method. Now the
resulting new parameter has the first parameter's frequency and offset::

    >>> a = P('A', frequency=2, offset=0.123)
    >>> new.get_derived([a])
    NewParameter('New Parameter', 2.0, 0.123)


This next block demonstrates how all parameters are aligned to the first available::

    >>> class NewParameter(Node):
    ...     def derive(self, a=P('A'), b=P('B'), c=P('C')):
    ...         print 'A frequency:%.2f offset:%.2f' % (a.frequency, a.offset) if a else 'A'
    ...         print 'B frequency:%.2f offset:%.2f' % (b.frequency, b.offset)
    ...         print 'C frequency:%.2f offset:%.2f' % (c.frequency, c.offset)

    >>> new = NewParameter()
    >>> a = P('A', frequency=2, offset=0.123)
    >>> b = P('B', frequency=4, offset=0.001)
    >>> c = P('C', frequency=0.25, offset=1.101)
    >>> new.get_derived([a, b, c])
    A frequency:2.00 offset:0.12
    B frequency:2.00 offset:0.12
    C frequency:2.00 offset:0.12
    NewParameter('New Parameter', 2.0, 0.123)


When '**a**' is not avialable the parameters are aligned to '**b**':

    >>> new.get_derived([None, b, c])
    A
    B frequency:4.00 offset:0.00
    C frequency:4.00 offset:0.00
    NewParameter('New Parameter', 4.0, 0.001)


Forcing Frequency and Offset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes up-sampling all dependencies to a higher frequency can be
beneficial to improve the accuracy of a derived parameter.::

    class NewParameter(Node):
        align_frequency = 4  #  Hz

Another useful feature is to force the offset, which is quite handy for
Flight Phases.::

    class NewParameter(Node):
        align_offset = 0


Turning off alignment
~~~~~~~~~~~~~~~~~~~~~

Aligning can be turned off, which means that one needs to account for the
dependencies having different frequencies and offsets.::

    class NewParameter(Node):
        align = False

The Node will default to the first available dependency's frequency and
offset. The typical use-case for not aligning parameters is when performing
customised merging of upsampling of the dependencies. In which case, it is
common to see the resulting frequency and offset being set on the class
within the derive method.::

    class NewParameter(Node):
        align = False
        def derive(self, a=P('A'), b=P('B')):
            # merge two signals
            self.array = merge(a, b)
            # set frequency and offset to be the average of a and b
            self.frequency = (a.frequency + b.frequency) / 2
            self.offset = (a.offset + b.offset) / 2

Node naming convention
~~~~~~~~~~~~~~~~~~~~~~

The following rules should be applied to Node names to ensure consistency:

* During Climb - This refers to any period when the aircraft is Climbing.
* During Descent - This refers to Descending periods between the Top Of Descent and Landing.
* While Climbing - This refers to Climbing periods between takeoff and Top Of Climb.
* While Descending - This refers to any period when the aircraft is Descending.