package org.dora.algorithm.solution;

/**
 * @author liulu
 * @date 2019-04-15
 */
public class SwordToOffer {

    /**
     * 二维数组重的查找
     *
     * @param target
     * @param array
     * @return
     */
    public boolean find(int target, int[][] array) {
        if (array == null || array.length == 0) {
            return false;
        }
        int row = array.length;
        int column = array[0].length;
        int i = row - 1;
        int j = 0;
        while (i >= 0 && j < column) {
            if (array[i][j] == target) {
                return true;
            } else if (array[i][j] < target) {
                j++;
            } else {
                i--;
            }
        }
        return false;
    }
}
