# Colorímetro por Foto (colorimetro-foto)

Abrir no navegador: [mborille.github.io/colorimetro-foto/colorimetro-foto.html](https://mborille.github.io/colorimetro-foto/colorimetro-foto.html)

Versão arquivada e citável: [DOI 10.6084/m9.figshare.34702197](https://doi.org/10.6084/m9.figshare.34702197)

Estima a cor do dente (CIELAB, D65) a partir de uma foto com cartão cinza 18% na mesma cena e calcula o ΔE00 contra uma cor desejada.

## Arquivos

- `colorimetro-foto.html`: app de um arquivo só. Abra no navegador (computador ou celular). A foto não sai do aparelho.
- `colorimetro_foto.py`: mesma matemática em Python (numpy e Pillow), para lote de fotos e estudos.
- `test_colorimetro_foto.py`: testes com os vetores oficiais do CIEDE2000 (Sharma, Wu e Dalal, 2005) e com foto sintética sob luz colorida.

## Método

1. A foto é lida como sRGB e convertida para RGB linear.
2. A média aparada do cartão (por padrão 10% de cada ponta de luminância) define um ganho por canal que leva o cartão a um cinza neutro com luminância igual à refletância nominal (0,18). Isso corrige balanço de branco e exposição de uma vez.
3. O mesmo ganho é aplicado ao dente. Em seguida: XYZ (D65, 2 graus) e CIELAB.
4. O ΔE00 usa CIEDE2000. Limiares de Paravina et al., J Esthet Restor Dent 2015;27 Suppl 1:S1-9: 0,8 (perceptibilidade) e 1,8 (aceitabilidade), 50:50.

O app e o Python dão o mesmo L*a*b* (diferença da ordem de 1e-11 na mesma imagem).

## Uso em Python

```
python colorimetro_foto.py foto.jpg --cartao 80,130,140,140 --dente 330,130,140,140 --alvo 72,3,20 --caso "Caso 014" --elemento 11 --csv medicoes.csv
```

As caixas são x, y, largura, altura em pixels da imagem. Várias fotos com as mesmas caixas podem ser passadas de uma vez.

## Limites (declarar em qualquer publicação)

É um estimador por câmera. Curva de tons do JPEG, perfil de cor, reflexo especular e luz desigual entre cartão e dente limitam a exatidão. Valide contra um espectrofotômetro antes de usar os números como dado de pesquisa. O programa não traz coordenadas de nenhuma escala comercial: a cor-alvo é digitada, lida de uma tabela no formato shade-table do dentcolor ou marcada na própria foto.

Licença MIT.

## Verificação

`python test_colorimetro_foto.py` (com `PYTHONPATH=.`) roda quatro testes: onze vetores de referência do CIEDE2000 (Sharma, Wu e Dalal, 2005), ida e volta CIELAB e sRGB, recuperação de cor em foto sintética sob quatro condições de luz e o alerta de cartão estourado. Na mesma imagem, o app e o módulo Python concordam no L*a*b* até a ordem de 1e-11.

## Como citar

Veja `CITATION.cff`.
