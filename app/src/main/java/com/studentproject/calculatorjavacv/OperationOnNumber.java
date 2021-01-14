package com.studentproject.calculatorjavacv;

import java.util.ArrayList;
import java.util.List;

public class OperationOnNumber
{
    private List<Integer> Dividers(int number)
    {
        List<Integer> dividers = new ArrayList<>();
        for (int i = number; i > 0; i--)
        {
            if (number % i == 0)
                dividers.add(i);
        }
        return dividers;
    }

    private List<Integer> Factors(int number)
    {
        List<Integer> factors = new ArrayList<>();
        for (int i = 2; i <= number; i++)
        {
            if (number % i == 0 && number > 1)
            {
                factors.add(i);
                number = number / i;
                i = 1;
            }
        }
        return factors;
    }

    private boolean isPrimeNumber(int number)
    {
        int counter = 0;
        for (int i = number; i > 1; i--)
        {
            if (number % i == 0)
                counter++;
            if (counter > 1)
                return false;
        }
        return true;

    }

    private boolean isOddNumber(int number)
    {
        if (number % 2 != 0)
            return true;
        else
            return false;
    }

}
