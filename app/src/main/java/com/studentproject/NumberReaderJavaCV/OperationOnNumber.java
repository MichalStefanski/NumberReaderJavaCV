package com.studentproject.NumberReaderJavaCV;

import java.util.ArrayList;
import java.util.List;

public class OperationOnNumber
{
    private List<Integer> Dividers(int number)
    {
        List<Integer> dividers = new ArrayList<>();
        for (int i = 1; i <= number; i++)
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
        return number % 2 != 0;
    }

    private int convertStringToNumber(String text)
    {
        int detectedNumber = 0;
        try
        {
            detectedNumber = Integer.parseInt(text);
        }
        catch (Exception e)
        {

        }
        return detectedNumber;
    }

    public String getDividers(String text)
    {
        List<Integer> temp = Dividers(convertStringToNumber(text));
        StringBuilder message = new StringBuilder("Dividers: ");
        for (int i = 0; i < temp.size(); i++)
        {
            if (i + 1 == temp.size())
            {
                message.append(temp.get(i));
            }
            else
            {
                message.append(temp.get(i));
                message.append(", ");
            }
        }
        return message.toString();
    }

    public  String getFactors(String text)
    {
        List<Integer> temp = Factors(convertStringToNumber(text));
        StringBuilder message = new StringBuilder("Factors: ");
        for (int i = 0; i < temp.size(); i++)
        {
            if (i + 1 == temp.size())
            {
                message.append(temp.get(i));
            }
            else
            {
                message.append(temp.get(i));
                message.append(" * ");
            }
        }
        return message.toString();
    }

    public String getIsOddNumber(String text)
    {

        return "Is number odd: " + isOddNumber(convertStringToNumber(text));
    }

    public  String getIsPrimeNumber(String text)
    {

        return "Is prime number: " + (isPrimeNumber(convertStringToNumber(text)));
    }
}
