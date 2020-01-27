package org.dora.algorithm.geeksforgeek;

/**
 * @author liulu12@xiaomi.com
 * @date 2020/1/27
 */
public class NumMatrix {


    private int[][] result;

    public NumMatrix(int[][] matrix) {
        result = matrix;
    }

    public int sumRegion(int row1, int col1, int row2, int col2) {
        if (result == null || result.length == 0) {
            return 0;
        }
        int ans = 0;
        for (int i = row1; i <= row2; i++) {
            for (int j = col1; j <= col2; j++) {
                ans += result[i][j];
            }
        }
        return ans;
    }

}
