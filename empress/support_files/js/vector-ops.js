/** @module vector utility-functions */
define([], function () {
    /**
     * Finds the angle of vector w.r.t the x-axis
     *
     * @param {Array} point - the point to find the angle for
     *
     * @return {Object}
     */
    function getAngle(point) {
        var x = point[0],
            y = point[1];
        var cos = x / Math.sqrt(x * x + y * y);
        var sin = Math.sqrt(1 - cos * cos);

        return {
            cos: cos,
            sin: sin,
        };
    }

    /**
     * Finds the magnitude of the vector
     *
     * @param {Array} point - the vector to find the magnitude of
     *
     * @return {Number}
     */
    function magnitude(point) {
        var x = point[0],
            y = point[1];
        return Math.sqrt(x * x + y * y);
    }

    /**
     * Rotates the vector
     *
     * @param {Array} point (x, y) coordinates
     * @param {Number} angle The amount to rotate the vector
     * @param {Boolean} over if true rotate point in positve sine direction
     *                       if false rotate point in negative sine direction
     *
     * @return {Array}
     */
    function rotate(point, angle, over) {
        var cos = angle.cos;
        var sin = angle.sin;
        var x = point[0];
        var y = point[1];

        // rotate the point in the negative sine direction (i.e. beneath x axis)
        if (over) {
            sin = -1 * sin;
        }

        x = point[0];
        y = point[1];

        point[0] = cos * x + -1 * sin * y;
        point[1] = sin * x + cos * y;

        return point;
    }

    /**
     * Translates the vector
     *
     * @param {Number} x - the amount to move along x-axis
     * @param {Number} y - the amount to move along y-axis
     *
     * @return {Array}
     */
    function translate(point, x, y) {
        point[0] = point[0] + x;
        point[1] = point[1] + y;
        return point;
    }

    /**
     * Returns an Object describing the top-left, top-right, bottom-left, and
     * bottom-right coordinates of a "thick line" box connecting two points
     * specified by (x1, y1) and (x2, y2).
     *
     * The outputs of this function can be passed into
     * Empress._addTriangleCoords(), for reference.
     *
     * @param {Number} x1
     * @param {Number} y1
     * @param {Number} x2
     * @param {Number} y2
     * @param {Number} amount - Thickness of the box to be drawn
     *
     * @return {Object} corners - Contains keys tL, tR, bL, bR
     */
    function computeBoxCorners(x1, y1, x2, y2, amount) {
        var point = translate([x1, y1], -1 * x2, -1 * y2);

        // find angle/length of branch
        var angle = getAngle(point);
        var length = magnitude(point);
        var over = point[1] < 0;

        // find top left of box of thick line
        tL = [0, amount];
        tL = rotate(tL, angle, over);
        tL = translate(tL, x2, y2);

        tR = [length, amount];
        tR = rotate(tR, angle, over);
        tR = translate(tR, x2, y2);

        // find bottom point of thick line
        bL = [0, -1 * amount];
        bL = rotate(bL, angle, over);
        bL = translate(bL, x2, y2);

        bR = [length, -1 * amount];
        bR = rotate(bR, angle, over);
        bR = translate(bR, x2, y2);
        // Idea of returning this as an object instead of a 2-D array based on
        // https://stackoverflow.com/questions/2917175/return-multiple-values-in-javascript#comment2969172_2917186
        return {tL: tL, tR: tR, bL: bL, bR: bR};
    }

    return {
        getAngle: getAngle,
        magnitude: magnitude,
        rotate: rotate,
        translate: translate,
        computeBoxCorners,
    };
});
