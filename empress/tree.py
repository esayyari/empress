# ----------------------------------------------------------------------------
# Copyright (c) 2016-2020, empress development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import warnings
from skbio import TreeNode
import numpy as np
from bp import BP, from_skbio_treenode


class TreeFormatWarning(Warning):
    pass


class Tree:
    """
    Attributes
    ----------
    length
    leafcount
    height
    depth

    Notes
    -----
    `length` refers to the branch length of a node to its parent.
    `leafcount` is the number of tips within a subtree. `height` refers
    to the longest path from root to the deepst leaf in that subtree.
    `depth` is the number of nodes found in the longest path.
    """

    def __init__(self, bp_tree):
        """ Constructs a Dendrogram object for visualization.

        Parameters
        ----------
        bp_tree: bp.BP
            BP tree object

        Returns
        -------

        """
        self.bp_tree = bp_tree
        self.B = self.bp_tree.B
        self.leafcounts = np.zeros(self.B.size, np.int)
        self.depths = np.zeros(self.B.size, np.double)
        self.heights = np.zeros(self.B.size, np.double)
        self.yr = np.zeros(self.B.size, np.double)
        self.xr = np.zeros(self.B.size, np.double)
        self.highest_child_yr = np.zeros(self.B.size, np.float)
        self.lowest_child_yr = np.zeros(self.B.size, np.float)
        self.clangle = np.zeros(self.B.size, np.double)
        self.clradius = np.zeros(self.B.size, np.double)
        self.xc0 = np.zeros(self.B.size, np.double)
        self.yc0 = np.zeros(self.B.size, np.double)
        self.xc1 = np.zeros(self.B.size, np.double)
        self.yc1 = np.zeros(self.B.size, np.double)
        self.highest_child_clangle = np.zeros(self.B.size, np.float)
        self.lowest_child_clangle = np.zeros(self.B.size, np.float)
        self.arcx0 = np.zeros(self.B.size, np.double)
        self.arcy0 = np.zeros(self.B.size, np.double)
        self.arcx1 = np.zeros(self.B.size, np.double)
        self.arcy1 = np.zeros(self.B.size, np.double)
        self.x1 = np.zeros(self.B.size, np.double)
        self.y1 = np.zeros(self.B.size, np.double)
        self.x2 = np.zeros(self.B.size, np.double)
        self.y2 = np.zeros(self.B.size, np.double)
        self.angle = np.zeros(self.B.size, np.double)
        self.childRem = -1

    @classmethod
    def from_tree(cls, tree, use_lengths=True):
        """ Creates an Tree object from a skbio tree.

        Parameters
        ----------
        tree : skbio.TreeNode
            Input skbio tree
        use_lengths: Boolean
            Specify if the branch length should be incorporated into
            the geometry calculations for visualization.
        Returns
        -------
        Tree: bp.BP

        """
        bp_tree = from_skbio_treenode(tree)
        if sum(bp_tree.B) <= 1:
            raise ValueError("Tree must contain at least 2 nodes.")

        # While traversing the tree, record tip / internal node names
        # (Nodes without names are ignored, since we'll assign those later
        # using tools.fill_missing_node_names())
        tip_names = []
        internal_node_names = []
        max_branch_length = 0
        for i in range(sum(bp_tree.B)):
            node_idx = bp_tree.postorderselect(i)
            name = bp_tree.name(node_idx)
            length = bp_tree.length(node_idx)
            if name is not None:
                # NOTE: This should eventually be taken out when
                # fill_missing_node_names() is refactored. However, for now,
                # this makes sure that users can't accidentally break things by
                # naming nodes identical to our default names for missing nodes
                if name.startswith("EmpressNode"):
                    raise ValueError(
                        'Node names can\'t start with "EmpressNode".'
                    )
                if bp_tree.isleaf(node_idx):
                    tip_names.append(name)
                else:
                    internal_node_names.append(name)
            if length is None:
                raise ValueError(
                    "Non-root branches of the tree must have lengths."
                )
            if length < 0:
                raise ValueError(
                    "Non-root branches of the tree must have nonnegative "
                    "lengths."
                )

            max_branch_length = max(length, max_branch_length)

        # We didn't consider the root node in the above traversal since we
        # don't care about its length. However, we do care about its name,
        # so we add the root's name to internal_node_names.

        if max_branch_length == 0:
            raise ValueError(
                "At least one non-root branch of the tree must have a "
                "positive length."
            )
        unique_tip_name_set = set(tip_names)
        if len(unique_tip_name_set) != len(tip_names):
            raise ValueError("Tip names in the tree must be unique.")

        unique_internal_node_name_set = set(internal_node_names)
        if len(unique_tip_name_set & unique_internal_node_name_set) > 0:
            raise ValueError(
                "Tip names in the tree cannot overlap with internal node "
                "names."
            )

        if len(unique_internal_node_name_set) != len(internal_node_names):
            warnings.warn(
                "Internal node names in the tree are not unique.",
                TreeFormatWarning
            )
        bp_tree.__class__ = Tree
        bp_tree.update_geometry(use_lengths)
        return bp_tree

    def postorder(self, include_self=True):
        e = sum(self.B) if include_self else sum(self.B) - 1
        for i in range(e):
            node_idx = self.bp_tree.postorderselect(i)
            yield node_idx

    def preorder(self, include_self=True):
        s = 0 if include_self else 1
        for i in range(s, sum(self.B)):
            node_idx = self.bp_tree.postorderselect(i)
            yield node_idx

    def bp_tree_tips(self):
        """ Extracts tip names in the tree, ignoring unnamed tips.

        Parameters
        ----------
        bp_tree : bp.BP
            Input BP tree
        Returns
        -------
        tips : list of strings
            list of tip names in the tree
        """
        tips = []
        # Iterate through all open and closing parentheses and extract tip names
        for i in range(self.B.size):
            pos_name = self.bp_tree.name(i)
            # Check if this is a leaf node with a label
            if self.isleaf(i) and (pos_name is not None):
                tips.append(pos_name)
        return tips

    def bp_tree_non_tips(self):
        """ Extracts internal node names in the tree, ignoring unnamed nodes.

           Parameters
           ----------
           bp_tree : bp.BP
               Input BP tree
           Returns
           -------
           non_tips : list of strings
               list of internal node names in the tree
        """
        non_tips = []
        for i in range(self.B.size):
            pos_name = self.bp_tree.name(i)
            # Check if this is an opening parenthesis, is not a leaf, and
            # has a node label
            if self.B[i] and not self.isleaf(i) and pos_name is not None:
                non_tips.append(pos_name)
        return non_tips

    def update_geometry(self, use_lengths, depth=None):
        """Calculate tree node attributes such as height and depth.

        Parameters
        ----------
        use_lengths: bool
            Specify if the branch length should be incorporated into
            the geometry calculations for visualization.
        depth: int
            The number of nodes in the longest path from root to leaf.
            This is agnostic to scale and orientation.

        """
        new_heights = np.zeros(self.B.size, dtype=np.double)
        new_leaf_count = np.zeros(self.B.size, dtype=np.int)
        new_depths = np.zeros(self.B.size, dtype=np.double)
        for node_idx in self.postorder():
            length = self.bp_tree.length(node_idx)
            if length is None or not use_lengths:
                if not use_lengths:
                    if self.isleaf(node_idx):
                        length = 5
                    else:
                        length = 1
                else:
                    length = 0
            new_depths[node_idx] = (depth or 0) + length

            if self.isleaf(node_idx):
                new_heights[node_idx] = length
                new_leaf_count[node_idx] = 1
            else:
                idx = self.bp_tree.fchild(node_idx)
                height = 0
                leafcount = 0
                while idx:
                    height = max(height, new_heights[idx])
                    leafcount += new_leaf_count[idx]
                    idx = self.bp_tree.nsibling(idx)
                height += length
                new_heights[node_idx] = height
                new_leaf_count[node_idx] = leafcount
        self.leafcounts = new_leaf_count
        self.heights = new_heights
        self.depths = new_depths

    def coords(self, height, width):
        """ Computes the coordinates of nodes to be rendered in plot.

        This runs multiple layout algorithms and saves all of the resulting
        coordinates for each node, so that layout algorithms can be rapidly
        toggled between in the JS interface.

        Also adds on .highest_child_yr and .lowest_child_yr attributes to
        internal nodes so that vertical bars for these nodes can be drawn in
        the rectangular layout.

        Parameters
        ----------
        height : int
            The height of the canvas.
        width : int
            The width of the canvas.

        Returns
        -------
        dict:
            Mapping between layout and the coordinate suffix.
        str:
            Name of the default layout.
        """

        layout_to_coordsuffix = {}
        layout_algs = (
            self.layout_unrooted,
            self.layout_rectangular,
            self.layout_circular,
        )
        # We set the default layout to whatever the first layout in
        # layout_algs is, but this behavior is of course modifiable
        default_layout = None
        for alg in layout_algs:
            name, suffix = alg(width, height)
            layout_to_coordsuffix[name] = suffix
            self.alter_coordinates_relative_to_root(suffix)
            if name == "Circular":
                self.alter_coordinates_relative_to_root("c0")
            if default_layout is None:
                default_layout = name

        # Determine highest and lowest child y-position for internal nodes in
        # the rectangular layout; used to draw vertical lines for these nodes.
        #
        # NOTE / TODO: This will have the effect of drawing vertical lines even
        # for nodes with only 1 child -- in this case lowest_child_yr ==
        # highest_child_yr for this node, so all of the stuff drawn in WebGL
        # for this vertical line shouldn't show up. I don't think this should
        # cause any problems, but it may be worth detecting these cases and not
        # drawing vertical lines for them in the future.
        for node_idx in self.preorder():
            if not self.isleaf(node_idx):
                # wow, child does not look like a word any more
                self.highest_child_yr[node_idx] = float("-inf")
                self.lowest_child_yr[node_idx] = float("inf")
                for c_idx in self.children(node_idx):
                    if self.yr[c_idx] > self.highest_child_yr[node_idx]:
                        self.highest_child_yr[node_idx] = self.yr[c_idx]
                    if self.yr[c_idx] < self.lowest_child_yr[node_idx]:
                        self.lowest_child_yr[node_idx] = self.yr[c_idx]

        return layout_to_coordsuffix, default_layout

    def alter_coordinates_relative_to_root(self, suffix):
        """ Subtracts the root node's x- and y- coords from all nodes' coords.

        This was previously done within coords(), but I moved it here so that
        this logic can be used after arbitrary layout computations.

        Parameters
        ----------
        suffix : str
            The suffix of the x- and y-coordinates to adjust.

            For example, this is "2" for the unrooted layout since coordinates
            are stored in the x2 and y2 attributes for every node; and it's "r"
            for the rectangular layout since the coordinate attributes are now
            xr and yr.
        """

        xname = "x" + suffix
        yname = "y" + suffix

        centersX = getattr(self, xname)
        centersY = getattr(self, yname)

        centerX = centersX[0]
        centerY = centersY[0]

        for node_idx in self.postorder():
            # This code might look sort of intimidating, but it's really just
            # another way to write out:
            #     node.x2 = node.x2 - centerX
            #     node.y2 = node.y2 - centerY
            # ...when we don't know what "x2" or "y2" will be named beforehand.
            centersX[node_idx] = centersX[node_idx] - centerX
            centersY[node_idx] = centersY[node_idx] - centerY
        setattr(self, xname, centersX)
        setattr(self, yname, centersY)

    def isleaf(self, i):
        """ Checks if node at position i belongs to a leaf node or not

            Parameters
           ----------
           bp_tree : bp.BP
               Input BP tree
            i : int
               The query node index
           Returns
           -------
           bool
               True if this is a leaf node, False otherwise
        """
        return self.B[i] and (not self.B[i + 1])

    def children(self, i):
        children = []
        child = self.bp_tree.fchild(i)
        while child > 0:
            children.append(child)
            child = self.bp_tree.nsibling(child)
        return children

    def layout_rectangular(self, width, height):
        """ Rectangular layout.

        In this sort of layout, each tip has a distinct y-position, and parent
        y-positions are centered over their descendant tips' positions.
        x-positions are computed based on nodes' branch lengths.

        Following this algorithm, nodes' rectangular layout coordinates are
        accessible at [node].xr and [node].yr.

        For a simple tree, this layout should look something like:
                 __
             ___|
         ___|   |__
        |   |___
        |    ___
        |___|
            |___

        Parameters
        ----------
        width : float
            width of the canvas
        height : float
            height of the canvas

        References
        ----------
        https://rachel53461.wordpress.com/2014/04/20/algorithm-for-drawing-trees/
            Clear explanation of Reingold-Tilford that I used a lot
        https://github.com/qiime/Topiary-Explorer/blob/master/src/topiaryexplorer/TreeVis.java
            Derived from the "Rectangular" layout algorithm code.
        """
        # NOTE: This doesn't draw a horizontal line leading to the root "node"
        # of the graph. See https://github.com/biocore/empress/issues/141 for
        # context.
        max_width = 0
        max_height = 0
        prev_y = 0
        for node_idx in self.postorder():
            if self.isleaf(node_idx):

                self.yr[node_idx] = prev_y
                prev_y += 1
                if self.yr[node_idx] > max_height:
                    max_height = self.yr[node_idx]
            else:
                # Center internal nodes above their children
                # We could also center them above their tips, but (IMO) this
                # looks better ;)
                children = self.children(node_idx)
                self.yr[node_idx] = sum([self.yr[c_idx] for
                                         c_idx in children]) / len(children)

        for node_idx in self.preorder(include_self=False):
            self.xr[node_idx] = self.xr[self.bp_tree.parent(node_idx)] + \
                                self.bp_tree.length(node_idx)
            if self.xr[node_idx] > max_width:
                max_width = self.xr[node_idx]

        # We don't check if max_width == 0 here, because we check when
        # constructing an Empress tree that it has at least one positive
        # branch length and no negative branch lengths. (And if this is the
        # case, then max_width must be > 0.)
        x_scaling_factor = width / max_width

        if max_height > 0:
            # Having a max_height of 0 could actually happen, in the funky case
            # where the entire tree is a straight line (e.g. A -> B -> C). In
            # this case our "rectangular layout" drawing places all nodes on
            # the same y-coordinate (0), resulting in max_height = 0.
            # ... So, that's why we only do y-scaling if this *isn't* the case.
            y_scaling_factor = height / max_height
        else:
            # Since this will be multiplied by 0 for every node, we can set
            # this to any real number and get the intended "effect" of keeping
            # every node's y-coordinate at 0.
            y_scaling_factor = 1

        for node_idx in self.preorder():
            self.xr[node_idx] *= x_scaling_factor
            self.yr[node_idx] *= y_scaling_factor

        # Now we have the layout! In the JS we'll need to draw each internal
        # node as a vertical line ranging from its lowest child y-position to
        # its highest child y-position, and then draw horizontal lines from
        # this line to all of its child nodes (where the length of the
        # horizontal line is proportional to the node length in question).
        return "Rectangular", "r"

    def layout_circular(self, width, height):
        """ Circular layout version of the rectangular layout.

        Works analogously to the rectangular layout:

            -Each tip is assigned a unique angle from the "center"/root of
             the tree (out of the range [0, 2pi] in radians), and internal
             nodes are set to an angle equal to the average of their
             children's. This mirrors the assignment of y-coordinates for
             the rectangular layout.

            -All nodes are then assigned a radius equal to the sum of their
             branch lengths descending from the root (but not including
             the root's branch length, if provided -- the root is represented
             as just a single point in the center of the layout). This mirrors
             the assignment of x-coordinates for the rectangular layout.

            -Lastly, we'll draw arcs for every internal node (except for the
             root) connecting the "start points" of the child nodes of that
             node with the minimum and maximum angle. (These points should
             occur at the radius equal to the "end" of the given internal
             node.)
                We don't draw this arc for the root node because we don't draw
                the root the same way we do the other nodes in the tree:
                the root is represented as just a single point at the center
                of the layout. Due to this, there isn't a way to draw an arc
                from the root, since the root's "end" is at the same point as
                its beginning (so the arc wouldn't be visible).

        Following this algorithm, nodes' circular layout coordinates are
        accessible at [node].xc and [node].yc. Angles will also be available
        at [node].clangle, and radii will be available at [node].clradius; and
        for non-root internal nodes, arc start and end coordinates will be
        available at [node].arcx0, [node].arcy0, [node].arcx1, & [node].arcy1.

        Parameters
        ----------
        width : float
            width of the canvas
        height : float
            height of the canvas

        References
        ----------
        https://github.com/qiime/Topiary-Explorer/blob/master/src/topiaryexplorer/TreeVis.java
            Description above + the implementation of this algorithm
            derived from the Polar layout algorithm code.
        """
        anglepernode = (2 * np.pi) / self.leafcounts[0]
        prev_clangle = 0
        for node_idx in self.postorder():
            if self.isleaf(node_idx):
                self.clangle[node_idx] = prev_clangle
                prev_clangle += anglepernode
            else:
                # Center internal nodes at an angle above their children
                children = self.children(node_idx)
                child_clangle_sum = sum([self.clangle[c_idx] for c_idx
                                         in children])
                self.clangle[node_idx] = child_clangle_sum / len(children)

        max_clradius = 0
        for node_idx in self.preorder(include_self=False):
            self.clradius[node_idx] = self.clradius[self.bp_tree.parent(node_idx)] + \
                                      self.bp_tree.length(node_idx)
            if self.clradius[node_idx] > max_clradius:
                max_clradius = self.clradius[node_idx]

        # Now that we have the polar coordinates of the nodes, convert these
        # coordinates to normal x/y coordinates.
        # NOTE that non-root nodes will actually have two x/y coordinates we
        # need to keep track of: one for the "end" of the node's line, and
        # another for the "start" of the node's line. The latter of these is
        # needed because the node's line begins at the parent node's radius but
        # the child node's angle, if that makes sense -- and since converting
        # from polar to x/y and back is annoying, it's easiest to just compute
        # this in python.
        max_x = max_y = float("-inf")
        min_x = min_y = float("inf")
        for node_idx in self.postorder():
            self.xc1[node_idx] = self.clradius[node_idx] * \
                                 np.cos(self.clangle[node_idx])
            self.yc1[node_idx] = self.clradius[node_idx] * \
                                 np.sin(self.clangle[node_idx])
            if self.isleaf(node_idx):
                # NOTE that the root has a clradius of 0 (since it's just
                # represented as a point at the center of the layout). We don't
                # even bother drawing the root in the Empress JS code, but for
                # the purposes of alter_coordinates_relative_to_root() we need
                # to explicitly position the root at (0, 0).
                self.xc0[node_idx] = 0
                self.yc0[node_idx] = 0
            else:
                self.xc0[node_idx] = self.clradius[
                                         self.bp_tree.parent(node_idx)] *\
                                     np.cos(self.clangle[node_idx])
                self.yc0[node_idx] = self.clradius[
                                         self.bp_tree.parent(node_idx)] *\
                                     np.sin(self.clangle[node_idx])
            # NOTE: We don't bother testing the xc0 / yc0 coordinates as
            # "extrema" because they should always be further "within" the
            # tree than the xc1 / yc1 coordinates.
            # TODO: verify that the "tree is a line" case doesn't mess this up.
            if self.xc1[node_idx] > max_x:
                max_x = self.xc1[node_idx]
            if self.yc1[node_idx] > max_y:
                max_y = self.yc1[node_idx]
            if self.xc1[node_idx] < min_x:
                min_x = self.xc1[node_idx]
            if self.yc1[node_idx] < min_y:
                min_y = self.yc1[node_idx]

        # TODO: raise error if the maximum and minimum are same for x or y.
        # may happen if the tree is a straight line.

        # set scaling factors
        # normalize the coordinate based on the largest dimension
        width_scale = width / (max_x - min_x)
        height_scale = height / (max_y - min_y)
        scale_factor = width_scale if width_scale > height_scale else \
            height_scale
        x_scaling_factor = scale_factor
        y_scaling_factor = scale_factor

        for node_idx in self.preorder():
            self.xc0[node_idx] *= x_scaling_factor
            self.yc0[node_idx] *= y_scaling_factor
            self.xc1[node_idx] *= x_scaling_factor
            self.yc1[node_idx] *= y_scaling_factor
            if not self.isleaf(node_idx) and (node_idx != 0):
                self.highest_child_clangle[node_idx] = float("-inf")
                self.lowest_child_clangle[node_idx] = float("inf")
                for c_idx in self.children(node_idx):
                    if self.clangle[c_idx] >\
                            self.highest_child_clangle[node_idx]:
                        self.highest_child_clangle[node_idx] =\
                            self.clangle[c_idx]
                    if self.clangle[c_idx] < \
                            self.lowest_child_clangle[node_idx]:
                        self.lowest_child_clangle[node_idx] =\
                            self.clangle[c_idx]
                # Figure out "arc" endpoints for the circular layout
                # NOTE: As with the "vertical lines" for internal nodes in the
                # rectangular layout, these arcs will be drawn for nodes with
                # only one child. Here, this case would mean that the
                # highest_child_clangle would equal the lowest_child_clangle,
                # so arcx0 would equal arcx1 and arcy0 would equal arcy1. So
                # nothing should show up (but it may be worth addressing this
                # in the future).
                self.arcx0[node_idx] = self.clradius[node_idx] * \
                                       np.cos(
                                           self.highest_child_clangle[node_idx])
                self.arcy0[node_idx] = self.clradius[node_idx] * \
                                       np.sin(
                                           self.highest_child_clangle[node_idx])
                self.arcx1[node_idx] = self.clradius[node_idx] * \
                                       np.cos(
                                           self.lowest_child_clangle[node_idx])
                self.arcy1[node_idx] = self.clradius[node_idx] * \
                                       np.sin(
                                           self.lowest_child_clangle[node_idx])
                self.arcx0[node_idx] *= x_scaling_factor
                self.arcy0[node_idx] *= y_scaling_factor
                self.arcx1[node_idx] *= x_scaling_factor
                self.arcy1[node_idx] *= y_scaling_factor

        return "Circular", "c1"

    def layout_unrooted(self, width, height):
        """ Find best scaling factor for fitting the tree in the figure.
        This method will find the best orientation and scaling possible to
        fit the tree within the dimensions specified by width and height, using
        an unrooted layout algorithm.

        Following this algorithm, nodes' unrooted layout coordinates are
        accessible at [node].x2 and [node].y2.

        Parameters
        ----------
        width : float
            width of the canvas
        height : float
            height of the canvas

        Returns
        -------
        best_scaling : float
            largest scaling factor in which the tree can fit in the canvas.

        Notes
        -----

        """
        # Recall that 360 degrees is equal to (2 * pi) radians.
        # You can think of this variable as "the maximum angle we can 'give' to
        # each leaf of the tree".
        angle = (2 * np.pi) / self.leafcounts[0]

        best_scale = 0
        for i in range(60):
            direction = i / 60.0 * np.pi

            (max_x, min_x, max_y, min_y) = self.update_unrooted_coords(
                1.0, 0, 0, direction, angle)

            x_diff = max_x - min_x
            width_min = 0
            if x_diff != 0:
                width_min = float(width) / x_diff
            y_diff = max_y - min_y
            height_min = 0
            if y_diff != 0:
                height_min = float(height) / y_diff
            scale = min(width_min, height_min)
            scale *= 0.95  # extra margin for labels
            if scale >= best_scale:
                best_scale = scale
                mid_x = width / 2 - ((max_x + min_x) / 2) * scale
                mid_y = height / 2 - ((max_y + min_y) / 2) * scale
                best_args = (scale, mid_x, mid_y, direction, angle)

        self.update_unrooted_coords(*best_args)
        return "Unrooted", "2"

    def update_unrooted_coords(self, s, x1, y1, a, da):
        """ Update x, y coordinates of tree nodes in canvas.

        This function will update the x1, y1, x2, y2, and angle attributes
        for all of the nodes within the tree. Note that (once the unrooted
        layout has finished) all that is really used are the x2 and y2
        attributes.

        In a server-based version of Empress, this could be applied when
        the tree becomes modified (i.e. pruning or collapsing) and the
        resulting coordinates would be modified to reflect the changes
        to the tree structure. (In practice, we just run this once on the
        Python side of things in order to precompute the layout.)

        Parameters
        ----------
        s : float
            scaling
        x1 : float
            x midpoint
        y1 : float
            y midpoint
        a : float
            angle (degrees)
        da : float
            angle resolution (degrees)

        Returns
        -------
        points : list of tuple
            2D coordinates of all of the nodes.
        """

        max_x = float('-inf')
        min_x = float('inf')
        max_y = float('-inf')
        min_y = float('inf')

        # calculates self coords/angle
        # Constant angle algorithm.  Should add maximum daylight step.
        x2 = x1 + self.bp_tree.length(0) * s * np.sin(a)
        y2 = y1 + self.bp_tree.length(0) * s * np.cos(a)
        (self.x1[0], self.y1[0], self.x2[0], self.y2[0], self.angle[0]) = \
            (x1, y1, x2, y2, a)
        node_indices = [node_idx for node_idx in
                        self.postorder(include_self=False)]
        node_indices.reverse()
        # for node in self.preorder(include_self=False):
        for node_idx in node_indices:
            x1 = self.x2[self.bp_tree.parent(node_idx)]
            y1 = self.y2[self.bp_tree.parent(node_idx)]

            # init a
            a = self.angle[self.bp_tree.parent(node_idx)]

            # same modify across nodes
            a = a - self.leafcounts[self.bp_tree.parent(node_idx)] * da / 2

            # check for conditional higher order
            for sib_idx in self.children(self.bp_tree.parent(node_idx)):
                if sib_idx != node_idx:
                    a += self.leafcounts[sib_idx] * da
                else:
                    a += (self.leafcounts[node_idx] * da) / 2
                    break

            # Constant angle algorithm.  Should add maximum daylight step.
            x2 = x1 + self.bp_tree.length(node_idx) * s * np.sin(a)
            y2 = y1 + self.bp_tree.length(node_idx) * s * np.cos(a)
            (self.x1[node_idx], self.y1[node_idx], self.x2[node_idx],
             self.y2[node_idx], self.angle[node_idx]) = (x1, y1, x2, y2, a)

            max_x, min_x = max(max_x, x2), min(min_x, x2)
            max_y, min_y = max(max_y, y2), min(min_y, y2)

        return (max_x, min_x, max_y, min_y)
