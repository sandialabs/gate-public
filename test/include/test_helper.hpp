#pragma once
#ifndef TEST_HELPER_HPP
#define TEST_HELPER_HPP

#include<gtest/gtest.h>
#include <vector>

inline void assertFloatArrayEq(std::vector<float> arr1, std::vector<float> arr2, float tol) {
	ASSERT_EQ(arr1.size(), arr2.size()) << "The two float arrays are not of equal size";
	for (int i = 0; i < arr1.size(); ++i) {
		ASSERT_NEAR(arr1[i], arr2[i], tol) << "Failed at index [i = " << i << "]" << std::endl;
	}
}

inline void assertFloatArrayEq(std::vector<float> arr1, std::vector<float> arr2, std::vector<float> tol) {
	ASSERT_EQ(arr1.size(), arr2.size()) << "The two float arrays are not of equal size";
	for (int i = 0; i < arr1.size(); ++i) {
		EXPECT_NEAR(arr1[i], arr2[i], tol[i]) << "Failed at index [i = " << i << "]" << std::endl;
	}
}

#endif // !TEST_HELPER_HPP