package org.example;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseException;
import com.github.javaparser.ast.CompilationUnit;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

public class Parser {
    public CompilationUnit useParser(File filename) throws IOException {
        InputStream in = null;
        CompilationUnit cu = null;
        try
        {
            in = new FileInputStream(filename);
            JavaParser javaParser = new JavaParser();
//            cu = new JavaParser().parse(in);
        }
//        catch(ParseException x)
//        {
//            // handle parse exceptions here.
//        }
        finally
        {
            in.close();
        }
        return cu;
    }
}

