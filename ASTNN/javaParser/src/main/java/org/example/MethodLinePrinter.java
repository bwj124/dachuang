package org.example;

import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import java.util.ArrayList;
import java.util.List;


public class MethodLinePrinter extends VoidVisitorAdapter {
    public List<Integer> allLines;

    public MethodLinePrinter(){
        this.allLines = new ArrayList<>();
    }

    @Override
    public void visit(MethodDeclaration md, Object arg) {
        allLines.add(md.getBegin().get().line);
    }

}
