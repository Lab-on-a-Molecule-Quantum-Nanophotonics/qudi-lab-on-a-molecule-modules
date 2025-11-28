import os
import importlib
from typing import Tuple
import tempfile

from docutils import nodes

from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective, SphinxRole
from sphinx.util.typing import ExtensionMetadata

from fysom import Fysom

class GraphDirective(SphinxDirective):
    """A directive to plot the graph of a state machine."""
    required_arguments = 1
    def run(self) -> list[nodes.Node]:
        qualifier = self.arguments[0]
        modulename,classname = qualifier.rsplit(sep='.', maxsplit=1)
        module = importlib.import_module(modulename)
        cls = getattr(module, classname)
        fsm = Fysom({
            'initial': cls._initial_state,
            'events': cls._SampledFiniteStateInterface__fysom_events_structure,
        })
        svgfile = None
        mapfile = None
        with tempfile.TemporaryDirectory() as tmpdirname:
            dotfilename=f"{qualifier}.dot"
            dotfilepath = os.path.join(tmpdirname, dotfilename)
            dotfile = open(dotfilepath, "w")
            dotfile.write(f"digraph statemachine_{classname}" + "{\n" + "    node[shape=rect];\n")# + 
            dotfile.write(f"    splines=ortho;\n")
            dotfile.write(f"    edge [lblstyle=\"above, sloped\"];\n")
            # graph = pygraphviz.AGraph(directed=True, id="statemachine_"+classname, name="statemachine_"+classname)
            for state in cls._SampledFiniteStateInterface__fysom_states:
                method = getattr(cls, state, None)
                label = state.replace("_", " ")
                tooltip=repr(method.__doc__).replace("'", "\"")
                dotfile.write(f"    {state} [label=\"{label}\", URL=\"#{modulename}.{qualifier}.{classname}.{state}\", tooltip={tooltip}];\n")
                # graph.add_node(state, 
                #                label=state.replace("_", " "), 
                #                tooltip=method.__doc__, 
                #                URL=f"#{qualifier}"
                # )
            for event, transitions in fsm._map.items():
                for from_state, to_state in transitions.items():
                    if from_state == "none":
                        from_state = "begin"
                    if from_state == "*":
                        for from_state in cls._SampledFiniteStateInterface__fysom_states:
                            # graph.add_edges_from([(from_state, to_state)], label=event)
                            dotfile.write(f"    {from_state} -> {to_state}[xlabel=\"{event}\"];\n")
                    else:
                        dotfile.write(f"    {from_state} -> {to_state}[xlabel=\"{event}\"];\n")
                        # graph.add_edges_from([(from_state, to_state)], label=event)

            dotfile.write("\n}")
            dotfile.close()
            mapfilepath = os.path.join(tmpdirname, f"{qualifier}.map")
            svgfilepath = os.path.join(tmpdirname, f"{qualifier}.svg")
            os.system(f"dot {dotfilepath} -Tcmapx -o {mapfilepath}")
            os.system(f"dot {dotfilepath} -Tsvg -o {svgfilepath}")
            with open(svgfilepath, "r") as f:
                svgfile = f.read()
            with open(mapfilepath, "r") as f:
                mapfile = f.read()
        image_node = nodes.raw('', svgfile, format='html')
        raw_node = nodes.raw('', mapfile, format='html')
        return [image_node, raw_node]

def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_directive('statemachine', GraphDirective)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
