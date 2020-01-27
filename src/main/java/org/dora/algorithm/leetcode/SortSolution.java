package org.dora.algorithm.leetcode;

/**
 * @author liulu12@xiaomi.com
 * @date 2019/12/13
 */
public class SortSolution {


    public static void main(String[] args) {
        SortSolution solution = new SortSolution();
        int[] nums = new int[]{5, 7, 1, -1, 4, 7};
        solution.quickSort(nums);

        for (int num : nums) {
            System.out.println(num);
        }

    }

    public int[] quickSort(int[] nums) {
        intervalQuickSort(nums, 0, nums.length - 1);
        return nums;
    }

    private void intervalQuickSort(int[] nums, int start, int end) {
        if (start > end) {
            return;
        }
        int partition = partition(nums, start, end);
        intervalQuickSort(nums, start, partition - 1);
        intervalQuickSort(nums, partition + 1, end);
    }

    private int partition(int[] nums, int left, int right) {
        int pivot = nums[left];
        while (left < right) {
            while (left < right && nums[right] >= pivot) {
                right--;
            }
            if (left < right) {
                nums[left] = nums[right];
                left++;
            }
            while (left < right & nums[left] <= pivot) {
                left++;
            }
            if (left < right) {
                nums[right] = nums[left];
                right--;
            }
        }
        nums[left] = pivot;
        return left;
    }
}
