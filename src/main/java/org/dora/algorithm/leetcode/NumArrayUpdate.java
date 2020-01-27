package org.dora.algorithm.leetcode;

/**
 * @author liulu12@xiaomi.com
 * @date 2020/1/27
 */
public class NumArrayUpdate {

    private int[] sum;


    public NumArrayUpdate(int[] nums) {
        sum = nums;
    }


    public void update(int i, int val) {
        sum[i] = val;
    }

    public int sumRange(int i, int j) {
        int result = 0;
        for (int index = i; i <= j; i++) {
            result += sum[index];
        }
        return result;
    }
}
