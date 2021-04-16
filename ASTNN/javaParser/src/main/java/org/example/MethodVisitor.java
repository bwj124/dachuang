package org.example;

import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

public class MethodVisitor extends VoidVisitorAdapter
{
    @Override
    public void visit(MethodDeclaration md, Object arg)
    {
        super.visit(md, arg);
        System.out.println("-" +md.getBegin().get().line+  "*");
    }
}
