package org.dora.algorithm.leetcode;

/**
 * @author dora
 * @date 2019/11/3
 */
public class Greedy {

    /**
     * 55. Jump Game
     *
     * @param nums
     * @return
     */
    public boolean canJump(int[] nums) {
        if (nums == null || nums.length == 0) {
            return false;
        }
        int reach = 0;

        for (int i = 0; i < nums.length - 1 && i <= reach; i++) {
            reach = Math.max(reach, i + nums[i]);
        }
        return reach >= nums.length - 1;
    }

    // --------卖煤气------//

    /**
     * 需要数学证明
     * 134. Gas Station
     */
    public int canCompleteCircuit(int[] gas, int[] cost) {
        if (gas == null || cost == null) {
            return 0;
        }
        int result = 0;
        int current = 0;
        int begin = 0;
        for (int i = 0; i < gas.length; i++) {
            current += gas[i] - cost[i];
            result += gas[i] - cost[i];
            if (current < 0) {
                begin = i + 1;
            }
        }
        return result <= 0 ? -1 : begin;
    }
}
